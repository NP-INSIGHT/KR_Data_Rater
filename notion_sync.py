"""
KR Chart Rater - Notion API 래퍼
Notion DB에서 종목 리스트 읽기 / 분석 보고서 쓰기
httpx로 직접 Notion API 호출 (안정적인 API 버전 사용)
"""

import time
import logging
import httpx
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
logger = logging.getLogger("kr_chart_rater")

NOTION_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


def _to_uuid(id_str):
    """32자 hex를 UUID 형식(하이픈 포함)으로 변환."""
    s = id_str.replace("-", "").strip()
    if len(s) == 32:
        return f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}"
    return id_str


class NotionSync:
    def __init__(self, token):
        self._token = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }
        self._last = 0

    def _throttle(self):
        """Notion rate limit: 3 req/sec"""
        gap = time.time() - self._last
        if gap < 0.35:
            time.sleep(0.35 - gap)
        self._last = time.time()

    def _post(self, path, body=None):
        """Notion API POST 요청 (429/5xx 자동 재시도)."""
        max_retries = 3
        for attempt in range(max_retries):
            self._throttle()
            resp = httpx.post(
                f"{NOTION_BASE}/{path}",
                headers=self._headers,
                json=body or {},
                timeout=None,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < max_retries - 1:
                    retry_after = int(resp.headers.get("Retry-After", 2 ** attempt * 5))
                    logger.warning(f"Notion API {resp.status_code}, retrying in {retry_after}s (attempt {attempt+1})")
                    time.sleep(retry_after)
                    continue
            if resp.status_code >= 400:
                logger.error(f"Notion API 오류: {resp.status_code} {resp.text}")
            resp.raise_for_status()
            return resp.json()

    def _get_db_schema(self, db_id):
        """DB 속성 이름 목록 조회."""
        self._throttle()
        resp = httpx.get(
            f"{NOTION_BASE}/databases/{db_id}",
            headers=self._headers,
            timeout=None,
        )
        if resp.status_code >= 400:
            logger.warning(f"DB 스키마 조회 실패: {resp.status_code}")
            return {}
        data = resp.json()
        schema = set(data.get("properties", {}).keys())
        logger.info(f"DB 스키마: {schema}")
        return schema

    # ── 종목 리스트 읽기 ──

    def read_watchlist(self, db_id, list_name=None):
        """종목 리스트 DB에서 종목명 읽기 (페이지네이션 포함).

        Args:
            db_id: Notion 데이터베이스 ID
            list_name: 필터링할 리스트 이름 (multi-select "리스트" 속성).
                       None이면 전체 종목 반환.
        """
        db_id = _to_uuid(db_id)
        names = []
        has_more = True
        start_cursor = None

        while has_more:
            body = {"page_size": 100}
            if start_cursor:
                body["start_cursor"] = start_cursor

            # 리스트 필터 (multi-select "리스트" 속성)
            if list_name:
                body["filter"] = {
                    "property": "리스트",
                    "multi_select": {"contains": list_name},
                }

            resp = self._post(f"databases/{db_id}/query", body)
            for page in resp.get("results", []):
                props = page.get("properties", {})

                # "종목명" title 추출
                title_prop = props.get("종목명", {})
                if title_prop.get("type") == "title":
                    titles = title_prop.get("title", [])
                    if titles:
                        name = titles[0].get("plain_text", "").strip()
                        if name:
                            names.append(name)

            has_more = resp.get("has_more", False)
            start_cursor = resp.get("next_cursor")

        logger.info(f"Notion 종목 리스트 읽기: {len(names)}개 종목 (DB: {db_id[:8]}..., 리스트: {list_name or '전체'})")
        return names

    # ── 보고서 쓰기 ──

    def create_report_page(self, db_id, title, date_str, summary_props, blocks):
        """보고서 DB에 새 페이지 생성."""
        db_id = _to_uuid(db_id)

        # DB 스키마 조회 → 존재하는 속성만 사용
        db_schema = self._get_db_schema(db_id)

        properties = {
            "날짜": {"title": [{"text": {"content": title}}]},
        }

        # 속성이 DB에 존재할 때만 추가
        prop_map = {
            "분석일": lambda v: {"date": {"start": v}},
            "종목수": lambda v: {"number": v},
            "선정": lambda v: {"number": v},
            "비용": lambda v: {"number": v},
            "리스트": lambda v: {"rich_text": [{"text": {"content": str(v)}}]},
            "A-1": lambda v: {"rich_text": [{"text": {"content": str(v)[:2000]}}]},
            "A-2": lambda v: {"rich_text": [{"text": {"content": str(v)[:2000]}}]},
        }

        for key, builder in prop_map.items():
            if key in summary_props and key in db_schema:
                properties[key] = builder(summary_props[key])

        # 페이지 생성 (최대 100 블록)
        first_blocks = blocks[:100]
        remaining_blocks = blocks[100:]

        page_body = {
            "parent": {"database_id": db_id},
            "properties": properties,
            "children": first_blocks,
        }
        page = self._post("pages", page_body)
        page_id = page["id"]

        # 100개 초과 블록은 append로 분할 추가
        while remaining_blocks:
            batch = remaining_blocks[:100]
            remaining_blocks = remaining_blocks[100:]
            self._post(f"blocks/{page_id}/children", {"children": batch})

        logger.info(f"Notion 보고서 페이지 생성: {title} (DB: {db_id[:8]}...)")
        return page_id

    def _chart_url(self, ticker_name, github_repo):
        """GitHub raw URL로 차트 이미지 URL 생성."""
        if not github_repo:
            return None
        from datetime import datetime
        date_str = datetime.now(KST).strftime("%Y%m%d")
        filename = f"{ticker_name}_{date_str}.png"
        return f"https://raw.githubusercontent.com/{github_repo}/main/charts/{filename}"

    def build_report_blocks(self, a_results, meta, github_repo=None):
        """A-1/A-2 종목 리스트를 Notion 블록으로 변환 (차트 이미지 + 선정 근거 포함)."""
        blocks = []

        total = meta.get("total_analyzed", 0)
        selected = len(a_results)
        provider = meta.get("provider", "")
        cost = meta.get("cost_usd", 0)

        data_basis = meta.get("data_basis", "")
        consensus_runs = meta.get("consensus_runs", 3)
        blocks.append(self._callout(
            f"분석 종목: {total}개 | 선정: {selected}개 | "
            f"LLM: {provider.upper()} | 합의: {consensus_runs}회 | 비용: ${cost:.4f}"
        ))
        if data_basis:
            blocks.append(self._paragraph(data_basis))

        if not a_results:
            blocks.append(self._paragraph("A-1/A-2 선정 종목 없음"))
            return blocks

        # A-1 / A-2 분리
        a1 = [r for r in a_results if r.get("grade") == "A-1"]
        a2 = [r for r in a_results if r.get("grade") == "A-2"]

        for grade_label, grade_results in [("A-1 (속도형 매력)", a1), ("A-2 (완만추세 지속형)", a2)]:
            if not grade_results:
                continue
            blocks.append(self._heading3(grade_label))
            for idx, r in enumerate(grade_results, 1):
                name = r.get("ticker_name", "")
                code = r.get("code", "")
                reliability = r.get("reliability", "")
                consensus = r.get("consensus_count", "")
                conf_str = f"신뢰도: {reliability}"
                if consensus:
                    conf_str += f", {consensus} 일치"
                blocks.append(self._heading3(f"{idx}. {name} ({code}) - {conf_str}"))

                # 등급 분포 (만장일치 아닌 경우)
                grade_dist = r.get("grade_distribution", {})
                if len(grade_dist) > 1:
                    dist_str = " / ".join(f"{g}: {c}회" for g, c in sorted(grade_dist.items()))
                    blocks.append(self._bulleted(f"등급 분포: {dist_str}"))

                # 차트 이미지
                chart_url = self._chart_url(name, github_repo)
                if chart_url:
                    blocks.append(self._image(chart_url))

                # 선정 근거
                reasoning = r.get("reasoning", "")
                if reasoning:
                    blocks.append(self._paragraph(reasoning))

                blocks.append(self._divider())

        blocks.append(self._paragraph(
            f"비용: ${cost:.4f} (약 {cost * 1400:.0f}원) | "
            f"Generated by KR Chart Rater"
        ))

        return blocks

    # ── Notion block 헬퍼 ──

    @staticmethod
    def _heading2(text):
        return {"type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": text}}]}}

    @staticmethod
    def _heading3(text):
        return {"type": "heading_3", "heading_3": {"rich_text": [{"type": "text", "text": {"content": text}}]}}

    @staticmethod
    def _paragraph(text):
        # Notion rich_text content 최대 2000자
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        return {"type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": c}} for c in chunks]}}

    @staticmethod
    def _bulleted(text):
        return {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": text}}]}}

    @staticmethod
    def _divider():
        return {"type": "divider", "divider": {}}

    @staticmethod
    def _callout(text):
        return {"type": "callout", "callout": {"rich_text": [{"type": "text", "text": {"content": text}}], "icon": {"type": "emoji", "emoji": "\U0001f4ca"}}}

    @staticmethod
    def _image(url):
        return {"type": "image", "image": {"type": "external", "external": {"url": url}}}

    @staticmethod
    def _bookmark(url):
        return {"type": "bookmark", "bookmark": {"url": url, "caption": []}}
