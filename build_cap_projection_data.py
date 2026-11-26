"""
2027~2031 샐러리캡(경쟁균형세) 예상 지출 데이터.

입력
- multi_year_contracts.csv          : 다년계약 구간 (옵션 연장 구간은 계약유형=option)
- offseason_moves.csv               : 2026 시즌 후 방출/이적 계약
- kbo_salary_out/salary_model/{snapshot}/
    panel.csv                       : 2026 소속, 제외사유(FA 자격 등)
    predictions_2027.csv            : 재계약 대상자 2027 예측 연봉 (train_salary_model.py --mode predict)
    team_cap_2026_detail.csv        : 2026 선수별 산정액, 예외선수 지정
    team_cap_2026_summary.csv       : 2026 구단별 상위 40인 합계
    profile_salary.csv              : 포지션, 나이

출력
- kbo_salary_out/cap_projection.json  (페이지 데이터)
- kbo_salary_cap_projection.html      (projection_template.html + 데이터 인라인. 스타일·구단 색은
                                       kbo_salary_cap_all_teams.html 에서 그대로 가져와 기존 대시보드와 같은 모양)

규칙
- 다년계약 연도별 산정액은 cap_contracts.py (규약 제184조):
    2026년 이전 체결 = 그 시즌 실제 연봉 + 계약금/보장연수 + 옵션/연수
    2026년 이후 체결 = (총액 - 옵션)/연수 + 옵션/연수   (옵션은 실지급액을 알 수 없어 균등 분배, 최대 지급 가정)
- 트레이드 선수의 계약은 현 소속 구단으로. 다른 구단으로 옮긴 선수의 옛 구단 옵션 연장 구간은 무효.
- 방출 선수의 잔여 구간은 뺀다.
- 예외선수(제185조)는 해마다 그 구단 7시즌 이상 소속 선수 중 산정액이 가장 큰 1명을 다시 골라 상한 계산에서만 50%.
- 2027 = 확정 계약 + 재계약 예측(상한 초과 시 인상분 축소된 값) 중 상위 40인.
  2028~2031 = 이미 계약된 금액만 (재계약·신규 FA 미포함).

사용법: python build_cap_projection_data.py [--snapshot 20261110]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from cap_contracts import (apply_opt_outs, clean_contracts, contract_cap_by_year, exception_tenure, known_salaries,
                           load_contracts, load_opt_outs, pick_exception, segment_schedule)

OUT_DIR = Path("kbo_salary_out")
MODEL_DIR = OUT_DIR / "salary_model"
YEARS = [2027, 2028, 2029, 2030, 2031]
# KBO 규약 제184조 (2026~2028 발표분). 2029년 이후는 미정 -> 페이지에서 연 5% 가정선으로만 표시
CAP_LIMIT = {2026: 1_439_723, 2027: 1_511_709, 2028: 1_587_294}
CAP_FLOOR = 606_538  # 제187조, 2027년부터
TEAMS = ["KIA", "KT", "LG", "NC", "SSG", "두산", "롯데", "삼성", "키움", "한화"]


def latest_snapshot() -> str:
    return sorted(p.name for p in MODEL_DIR.iterdir() if p.is_dir() and p.name.isdigit())[-1]


def render_page(data: dict):
    """projection_template.html 에 기존 대시보드(kbo_salary_cap_all_teams.html)의 CSS·구단 색과 데이터를 채운다."""
    base = Path("kbo_salary_cap_all_teams.html").read_text(encoding="utf-8")
    css = base[base.index("<style>") + len("<style>"):base.index("</style>")]
    tag = '<script id="team-meta-data" type="application/json">'
    meta = base[base.index(tag) + len(tag):]
    meta = meta[:meta.index("</script>")].strip()
    page = (Path(__file__).resolve().parent / "projection_template.html").read_text(encoding="utf-8")
    page = (page.replace("/*__BASE_CSS__*/", css)
                .replace("__TEAM_META__", meta)
                .replace("__PROJECTION_DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    Path("kbo_salary_cap_projection.html").write_text(page, encoding="utf-8")
    print("[저장] kbo_salary_cap_projection.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot")
    args = ap.parse_args()
    S = MODEL_DIR / (args.snapshot or latest_snapshot())

    panel = pd.read_csv(S / "panel.csv", dtype={"playerId": str}, encoding="utf-8-sig")
    p26 = panel[panel["시즌"] == 2026].copy()
    p26["팀"] = p26["팀"].astype(str).str.split("/").str[-1]
    team_of = dict(zip(p26["playerId"], p26["팀"]))
    name_of = dict(zip(p26["playerId"], p26["선수명"]))
    why_of = dict(zip(p26["playerId"], p26["제외사유"].fillna("")))

    prof = pd.read_csv(S / "profile_salary.csv", dtype={"playerId": str}, encoding="utf-8-sig")
    pos_of = dict(zip(prof["playerId"], prof["포지션상세"].fillna(prof["포지션군"])))
    age_of = {pid: (int(a) + 1 if pd.notna(a) else None) for pid, a in zip(prof["playerId"], prof["나이_2026"])}
    # 동명이인 구분용: 등번호, 생년
    no_of = {pid: (str(n).split(".")[0] if pd.notna(n) else None) for pid, n in zip(prof["playerId"], prof["등번호"])}
    by_of = {pid: (int(str(d)[:4]) if pd.notna(d) else None) for pid, d in zip(prof["playerId"], prof["생년월일"])}

    moves = pd.read_csv("offseason_moves.csv", dtype={"playerId": str})
    released = set(moves.loc[moves["구분"].str.contains("방출", na=False), "playerId"])
    m, fixes = clean_contracts(load_contracts("multi_year_contracts.csv"), team_of, released)

    det = pd.read_csv(S / "team_cap_2026_detail.csv", dtype={"playerId": str}, encoding="utf-8-sig")
    ex_ids = set(det.loc[det["예외선수지정"] == True, "playerId"])  # noqa: E712  (2026 지정자)
    tenure = exception_tenure(det)
    v26 = {(r.구단, r.playerId): float(r.산정액_옵션최대_만원) for r in det.itertuples()}
    s26 = pd.read_csv(S / "team_cap_2026_summary.csv", encoding="utf-8-sig").set_index("구단")

    pred = pd.read_csv(S / "predictions_2027.csv", dtype={"playerId": str}, encoding="utf-8-sig")
    pred["팀"] = pred["팀"].astype(str).str.split("/").str[-1]
    pred_of = {r.playerId: r for r in pred.itertuples()}

    # 선수 x 연도 셀: {"v": 산정액(만원), "k": G 보장 / O 옵션연장 / P 예측 / S 단년계약}
    cells: dict[str, dict] = {}
    meta: dict[str, dict] = {}

    def put(pid, team, year, v, k, end=None):
        meta.setdefault(pid, {"team": team})
        cells.setdefault(pid, {})[str(year)] = {"v": round(v), "k": k, **({"end": end} if end else {})}

    known = known_salaries(S, OUT_DIR / "friend_data_parsed.json", m)
    sched = apply_opt_outs(contract_cap_by_year(m, known, YEARS), m, load_opt_outs())
    for (pid, y), v in sorted(sched.items(), key=lambda kv: kv[0][1]):
        r = v["row"]
        kind = "D" if v["method"].startswith("옵트아웃") else ("O" if r.옵션구간 else "G")
        put(pid, r.현소속, y, v["hi"], kind, int(r.계약종료))
        c = cells[pid][str(y)]
        c["lo"] = round(v["lo"])
        c["m"] = v["method"]
        if v["method"] == "구방식":
            c["sal"], c["bon"], c["opt"] = v["salary"], round(v["bonus"]), round(v["option"])
            c["est"] = v["salary_src"].startswith("추정")
        else:
            c["opt"] = round(v["option"])
        name_of.setdefault(pid, r.선수명)
    for r in moves[moves["새연봉_만원"].notna()].itertuples():
        put(r.playerId, r.새구단, 2027, float(r.새연봉_만원), "S")
        meta[r.playerId]["team"] = r.새구단
        name_of.setdefault(r.playerId, r.선수명)
    for r in pred.itertuples():
        if "2027" not in cells.get(r.playerId, {}):
            put(r.playerId, r.팀, 2027, float(r.예측_보정_만원), "P")
            meta[r.playerId]["prev"] = float(r.연봉_만원)

    # 스크래핑 명단에 없던 수동 추가 선수 (roster_additions.csv): 모델 입력이 없어 다음 시즌은 동결 가정
    if Path("roster_additions.csv").exists():
        ad = pd.read_csv("roster_additions.csv", dtype={"playerId": str}, encoding="utf-8-sig")
        for r in ad[ad["시즌"] == 2026].itertuples():
            name_of[r.playerId] = r.선수명
            pos_of[r.playerId] = r.포지션군
            age_of[r.playerId] = 2027 - int(str(r.생년월일)[:4])
            by_of[r.playerId] = int(str(r.생년월일)[:4])
            team_of[r.playerId] = r.구단
            if r.다음시즌_가정 in ("동결", "배율") and "2027" not in cells.get(r.playerId, {}):
                k = float(r.다음시즌_배율) if r.다음시즌_가정 == "배율" and pd.notna(getattr(r, "다음시즌_배율", None)) else 1.0
                put(r.playerId, r.구단, 2027, float(r.연봉_만원) * k, "P")
                cells[r.playerId]["2027"]["asm"] = (f"명목 연봉 {int(r.연봉_만원):,}만원 x{k:g} 가정. " if k != 1 else "2026 연봉 동결 가정. ") + str(r.비고)
                meta[r.playerId]["prev"] = float(r.연봉_만원)

    # 계약 구간 정보 (페이지의 계약 정보 팝업·연봉/옵션 설정에 씀)
    #   fa : FA 계약(Y) / 비FA 다년계약(N)
    #   ys : 2026년 이전 체결(구방식) 계약의 연도별 연봉 {연도: [연봉 만원, 추정여부]} (지난 시즌 포함, 군복무 등 쉰 해는 없음)
    #   oo : 옵트아웃 가능 연도 (multi_year_contracts.csv 의 옵트아웃연도 열이 있으면)
    terms_of: dict[str, list] = {}
    has_oo = "옵트아웃연도" in m.columns
    for r in m[m["계약종료"] >= 2027].sort_values("계약시작").itertuples():
        term = {"s": int(r.계약시작), "e": int(r.계약종료), "n": int(r.계약총연수 or 1), "tot": float(r.계약총액_억원),
                "bon": float(r.계약금_억원), "opt": float(r.옵션총액_억원), "o": bool(r.옵션구간), "sg": int(r.계약체결일),
                "team": r.현소속, "fa": str(getattr(r, "FA여부", "")).strip().upper() == "Y"}
        sch = segment_schedule(r, known)
        ys = {str(y): [v["salary"], 1 if str(v["salary_src"] or "").startswith("추정") else 0]
              for y, v in sorted(sch.items()) if v["method"] == "구방식"}
        if ys:
            term["ys"] = ys
        if has_oo and pd.notna(getattr(r, "옵트아웃연도", None)) and str(r.옵트아웃연도).strip():
            term["oo"] = [int(x) for x in str(r.옵트아웃연도).replace("/", ",").split(",") if x.strip().isdigit()]
        terms_of.setdefault(r.playerId, []).append(term)

    # 다음 FA 취득 시즌 (fa_league_summary.csv 의 FA까지_남은시즌, 2026시즌 종료 기준):
    #   n시즌 더 뛰면 자격 -> 2027+n 시즌을 앞두고 FA. 계약이 더 길면 페이지에서 계약이 끝난 다음 시즌으로 미룬다.
    fa0_of = {}
    fa_path = OUT_DIR / "fa_league_summary.csv"
    if fa_path.exists():
        fs = pd.read_csv(fa_path, dtype={"playerId": str}, encoding="utf-8-sig")
        fa0_of = {r.playerId: 2027 + int(r.FA까지_남은시즌) for r in fs.itertuples() if pd.notna(r.FA까지_남은시즌)}

    teams_out = {}
    for t in TEAMS:
        ids = [pid for pid, mm in meta.items() if mm["team"] == t]
        # 2027에 이 팀에 남지 않는 2026 선수 (FA 자격 취득 / 방출 / 이적 / 웨이버)
        gone = []
        for r in det[det["구단"] == t].itertuples():
            pid = r.playerId
            if pid in meta and meta[pid]["team"] == t:
                continue
            why = why_of.get(pid, "")
            if pid in meta:
                st = f"{meta[pid]['team']} 이적"
            elif pid in released:
                st = "방출"
            elif "FA자격취득_2027" in why:
                st = "FA"
            elif "웨이버/미소속" in why or team_of.get(pid) not in (t, None):
                st = "이탈"
            else:
                continue  # 연봉 없음 등 판단 불가 -> 표에서 생략
            gone.append({"pid": pid, "name": r.선수명, "pos": pos_of.get(pid, ""), "age": age_of.get(pid),
                         "y2026": round(float(r.산정액_옵션최대_만원)), "status": st, "ex": pid in ex_ids,
                         "ten": tenure.get((pid, t), 0), "no": no_of.get(pid), "by": by_of.get(pid), "fa0": fa0_of.get(pid)})

        players = []
        for pid in ids:
            for c in cells.get(pid, {}).values():
                c.setdefault("lo", c["v"])
            players.append({"pid": pid, "name": name_of.get(pid, ""), "pos": pos_of.get(pid, ""), "age": age_of.get(pid),
                            "ex": pid in ex_ids, "y2026": round(v26[(t, pid)]) if (t, pid) in v26 else None,
                            "prev": meta[pid].get("prev"), "cells": cells.get(pid, {}),
                            "ten": tenure.get((pid, t), 0), "ct": terms_of.get(pid, []),
                            "no": no_of.get(pid), "by": by_of.get(pid), "fa0": fa0_of.get(pid)})

        years = {}
        for y in YEARS:
            ys = str(y)
            vals, lo_vals, parts = [], [], {"G": 0.0, "O": 0.0, "P": 0.0, "S": 0.0, "D": 0.0}
            ex_pid = pick_exception({pid: p["cells"][ys]["v"] for p, pid in zip(players, ids)
                                     if ys in p["cells"] and p["cells"][ys]["k"] != "D"},
                                    t, y, tenure)
            ent = []  # (cap 반영 비율 f, 옵션 0 금액, 옵션(인센티브) 몫, 구분)
            for p, pid in zip(players, ids):
                c = p["cells"].get(ys)
                if not c:
                    continue
                c["ex"] = pid == ex_pid
                f = 0.5 if c["ex"] else 1.0
                lo = c.get("lo", c["v"]) if c["k"] in ("G", "S") else c["v"]
                ent.append((f, lo, c["v"] - lo, c["k"]))
                lo_vals.append(c.get("lo", c["v"]))
            top = sorted(ent, key=lambda e: -e[0] * (e[1] + e[2]))[:40]
            # 옵션 실행률 기본값: 상한에 걸리지 않으면 100%, 걸리면 딱 상한까지만 (사용자 규칙 2026-11-26)
            cap_y = CAP_LIMIT.get(y, round(CAP_LIMIT[2028] * 1.05 ** (y - 2028)))
            base = sum(f * lo for f, lo, o, k in top)
            optsum = sum(f * o for f, lo, o, k in top)
            rate = 1.0 if base + optsum <= cap_y or optsum == 0 else max(0.0, min(1.0, (cap_y - base) / optsum))
            for f, lo, o, k in top:
                parts[k] += f * (lo + rate * o)
            top40 = sum(parts.values())
            years[ys] = {"top40": round(top40), "ex": name_of.get(ex_pid, "") if ex_pid else "", "opt_rate": round(rate, 4),
                         "guar": round(parts["G"] + parts["S"] + parts["D"]), "dead": round(parts["D"]), "opt": round(parts["O"]),
                         "pred": round(parts["P"]), "count": len(ent),
                         "contracted": sum(1 for p in players if p["cells"].get(ys, {}).get("k") in ("G", "O", "S")),
                         "floor_basis": round(sum(sorted(lo_vals, reverse=True)[:40]))}
        teams_out[t] = {"y2026": round(float(s26.loc[t, "상위40_옵션최대_예외적용"])),
                        "ex": str(s26.loc[t, "예외선수"]), "years": years,
                        "players": players, "gone": sorted(gone, key=lambda g: -g["y2026"])}

    # 신인 드래프트 지명 순서: N년 드래프트는 N-2년 최종순위의 역순, 1~11라운드 모두 같은 순서(스네이크 아님).
    # 순위가 나온 시즌까지만 순서를 알 수 있고, 그 다음 해 드래프트는 순서 미정(None).
    rank = pd.read_csv(S / "team_rank.csv", encoding="utf-8-sig")
    last = int(rank["연도"].max())
    draft = {}
    for dy in (last + 2, last + 3):
        r = rank[rank["연도"] == dy - 2]
        draft[str(dy)] = r.sort_values("최종순위", ascending=False)["팀명"].tolist() if len(r) == 10 else None
    out = {"snapshot": S.name, "years": YEARS, "cap": CAP_LIMIT, "floor": CAP_FLOOR,
           "fixes": fixes, "draft": draft, "draft_rounds": 11, "teams": teams_out}
    (OUT_DIR / "cap_projection.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[저장] {OUT_DIR / 'cap_projection.json'}")
    render_page(out)
    for f in fixes:
        print("  ·", f)
    for t in TEAMS:
        y = teams_out[t]["years"]
        print(f"  {t:4s} 2026 {teams_out[t]['y2026']/1e4:6.1f}억 | " +
              " ".join(f"{k} {v['top40']/1e4:6.1f}" for k, v in y.items()))


if __name__ == "__main__":
    main()
