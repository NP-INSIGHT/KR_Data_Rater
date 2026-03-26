"""
KR Data Rater - Notion 기반 완전자동화 분석 스크립트
GitHub Actions에서 매일 실행.

사용법:
  python run_notion.py                      # 전체 (분석 + 보고서)
  python run_notion.py --provider claude    # Claude 사용
  python run_notion.py --list "BIO"         # 특정 리스트만 필터
  python run_notion.py --dry-run            # 분석만, Notion 저장 안 함
  python run_notion.py --report-only        # 저장된 결과로 보고서만 작성
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import engine
from notion_sync import NotionSync


RESULTS_JSON = engine.BASE_DIR / "results" / "latest_results.json"


def _format_stock_list(results):
    """A-1 또는 A-2 결과를 '종목명(티커) 종가원 [신뢰도], ...' 형식으로 변환."""
    parts = []
    for r in results:
        price = r.get("last_close")
        reliability = r.get("reliability", "")
        base = f"{r['ticker_name']}({r.get('code', '')})"
        if price:
            base += f" {price:,}원"
        base += f" [{reliability}]"
        parts.append(base)
    return ", ".join(parts)


def _save_results(a_results, total_analyzed, total_errors, total_usage, provider, list_name):
    """분석 결과를 JSON으로 저장."""
    RESULTS_JSON.parent.mkdir(exist_ok=True)
    data = {
        "a_results": a_results,
        "total_analyzed": total_analyzed,
        "total_errors": total_errors,
        "total_usage": total_usage,
        "provider": provider,
        "list_name": list_name,
        "timestamp": datetime.now(engine.KST).isoformat(),
    }
    RESULTS_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_results():
    """저장된 분석 결과 로드."""
    if not RESULTS_JSON.exists():
        return None
    return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))


def _run_analysis(list_name, provider, notion, log):
    """종목 분석 수행 (Notion 저장 없이 결과만 반환/저장)."""

    notion_wl_db = engine.CONFIG.get("NOTION_WATCHLIST_DB", "")
    if not notion_wl_db:
        log("[X] NOTION_WATCHLIST_DB가 config.txt에 설정되지 않았습니다.")
        return

    log(f"Notion에서 종목 읽기... (리스트: {list_name or '전체'})")
    ticker_names = notion.read_watchlist(notion_wl_db, list_name=list_name)

    if not ticker_names:
        log("분석할 종목 없음, 종료합니다.")
        return

    log(f"\n{len(ticker_names)}개 종목 | LLM: {provider}")

    a_results = []
    total_usage = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "input_cost_usd": 0, "output_cost_usd": 0, "total_cost_usd": 0,
        "api_calls": 0,
    }
    total_analyzed = 0
    total_errors = 0

    for name in ticker_names:
        log(f"\n--- {name} ---")

        try:
            yf_ticker, code, market = engine.resolve_ticker(name)
            df, _, _ = engine.fetch_ohlcv(name)
        except Exception as e:
            log(f"  [X] {name} 데이터 오류: {e}")
            total_errors += 1
            continue

        try:
            # 데이터 기반 분석 (차트 이미지 대신 텍스트 데이터)
            data_text = engine.prepare_analysis_data(df, name, code)
            result = engine.analyze_with_consensus(
                data_text=data_text,
                ticker_name=name,
                provider=provider,
            )
            result["code"] = code
            result["market"] = market
            result["last_close"] = int(df["Close"].iloc[-1])

            # 보고서용 차트 생성
            if engine.SAVE_CHARTS:
                chart_path = engine.generate_chart(name, df, market=market)
                result["chart_path"] = str(chart_path)
            else:
                result["chart_path"] = ""

            grade = result.get("grade", "N/A")
            reliability = result.get("reliability", "")
            consensus_count = result.get("consensus_count", "")
            grade_dist = result.get("grade_distribution", {})
            dist_str = " / ".join(f"{g}:{c}" for g, c in sorted(grade_dist.items()))
            log(f"  [분석] {grade} (신뢰도: {reliability}, {consensus_count}, 분포: {dist_str})")

            usage = result.get("token_usage", {})
            if usage:
                engine._accumulate_usage(total_usage, usage)
            total_analyzed += 1

            if grade in ("A-1", "A-2"):
                # threshold 필터링
                conf = result.get("consensus_confidence", result.get("confidence", 0))
                agreement = int(consensus_count.split("/")[0]) if "/" in consensus_count else 0
                threshold = engine.CONFIDENCE_THRESHOLD
                min_agree = engine.MIN_CONSENSUS_AGREEMENT

                if agreement >= min_agree and conf >= threshold:
                    a_results.append(result)
                    log(f"  *** {grade} 선정 (신뢰도: {reliability}, {consensus_count}) ***")
                else:
                    log(f"  [FILTERED] {grade} 탈락: 신뢰도={reliability}, 합의={consensus_count} "
                        f"(기준: >={min_agree}회 일치)")

        except Exception as e:
            log(f"  [X] {name} 분석 오류: {e}")
            total_errors += 1

    # A-1 우선, A-2 다음, confidence 내림차순 정렬
    a_results.sort(key=lambda x: (0 if x.get("grade") == "A-1" else 1, -x.get("confidence", 0)))

    cost = total_usage["total_cost_usd"]
    log(f"\n{'='*50}")
    log(f"분석 완료: {total_analyzed}개 분석, {total_errors}개 오류")
    log(f"A-1/A-2 선정: {len(a_results)}개")
    log(f"비용: ${cost:.4f} (약 {cost*1400:.0f}원)")

    if a_results:
        log("\nA-1/A-2 선정 종목:")
        for i, r in enumerate(a_results, 1):
            log(f"  {i}. [{r.get('grade', '')}] {r['ticker_name']} ({r.get('code', '')})")

    _save_results(a_results, total_analyzed, total_errors, total_usage, provider, list_name)
    log(f"\n결과 저장: {RESULTS_JSON}")


def _run_report(notion, github_repo, log):
    """저장된 분석 결과로 Notion 보고서 작성."""

    notion_rp_db = engine.CONFIG.get("NOTION_REPORT_DB", "")
    if not notion_rp_db:
        log("[X] NOTION_REPORT_DB가 config.txt에 설정되지 않았습니다.")
        return

    data = _load_results()
    if not data:
        log("[X] 저장된 분석 결과가 없습니다. 먼저 분석을 실행하세요.")
        return

    a_results = data["a_results"]
    provider = data["provider"]
    list_name = data.get("list_name")
    total_analyzed = data["total_analyzed"]
    total_errors = data["total_errors"]
    cost = data["total_usage"]["total_cost_usd"]

    a1_results = [r for r in a_results if r.get("grade") == "A-1"]
    a2_results = [r for r in a_results if r.get("grade") == "A-2"]

    log(f"Notion 보고서 작성 중... (A-1: {len(a1_results)}개, A-2: {len(a2_results)}개)")

    try:
        dt = datetime.now(engine.KST)
        title = dt.strftime("%m/%d %H:%M")
        if dt.hour >= 16 or (dt.hour == 15 and dt.minute >= 30):
            data_basis = f"데이터 기준: {dt.strftime('%Y-%m-%d')} 마감봉 (당일)"
        else:
            data_basis = f"데이터 기준: 직전 거래일 마감봉 (장중 미반영)"
        meta = {
            "total_analyzed": total_analyzed,
            "a_count": len(a_results),
            "consensus_runs": engine.CONSENSUS_RUNS,
            "errors": total_errors,
            "provider": provider,
            "cost_usd": cost,
            "data_basis": data_basis,
        }
        summary_props = {
            "분석일": dt.strftime("%Y-%m-%d"),
            "리스트": list_name or "전체",
            "종목수": total_analyzed,
            "선정": len(a_results),
            "비용": round(cost, 4),
            "A-1": _format_stock_list(a1_results),
            "A-2": _format_stock_list(a2_results),
        }
        blocks = notion.build_report_blocks(a_results, meta, github_repo)
        notion.create_report_page(notion_rp_db, title, dt.strftime("%Y-%m-%d"), summary_props, blocks)
        log(f"Notion 보고서 작성 완료: {title}")
    except Exception as e:
        log(f"[X] Notion 보고서 작성 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="KR Data Rater - Notion 자동화 분석")
    parser.add_argument(
        "--provider", default=None, choices=["claude", "gemini"],
        help="LLM provider (기본: config.txt 설정값)",
    )
    parser.add_argument(
        "--list", default=None,
        help="Notion DB multi-select 필터 (예: 'BIO'). 미지정 시 전체.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="분석만 수행, Notion 저장 안 함",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="저장된 결과로 Notion 보고서만 작성 (분석 건너뜀)",
    )
    args = parser.parse_args()

    provider = args.provider or engine.CONFIG.get("LLM_PROVIDER", engine.LLM_PROVIDER)
    github_repo = engine.CONFIG.get("GITHUB_REPO", "")

    def log(msg):
        print(msg, flush=True)

    # Notion 클라이언트 초기화
    try:
        notion_token = engine.read_secret("notion_api_key.txt")
        notion = NotionSync(notion_token)
    except Exception as e:
        log(f"[X] Notion 인증 실패: {e}")
        log("NOTION_API_KEY 환경변수 또는 secrets/notion_api_key.txt를 확인하세요.")
        return 2

    if args.report_only:
        _run_report(notion, github_repo, log)
    elif args.dry_run:
        _run_analysis(args.list, provider, notion, log)
    else:
        _run_analysis(args.list, provider, notion, log)
        _run_report(notion, github_repo, log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
