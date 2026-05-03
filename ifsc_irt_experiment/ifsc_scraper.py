from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = 'https://components.ifsc-climbing.org'
logger = logging.getLogger(__name__)


class IFSCScraper:
    def __init__(
        self,
        raw_dir: Path,
        sleep_seconds: float = 1.5,
        max_retries: int = 5,
        timeout_seconds: int = 30,
        user_agent: str = 'ClimbingResultsResearch/0.1 (contact: your-email@example.com)',
    ) -> None:
        self.raw_dir = raw_dir
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})

    def _cache_key(self, params: dict[str, Any]) -> str:
        payload = json.dumps(params, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

    def _cache_path(self, params: dict[str, Any]) -> Path:
        api = str(params.get('api', 'unknown'))
        return self.raw_dir / f'{api}_{self._cache_key(params)}.json'

    def _decode_response(self, response: requests.Response) -> Any:
        content_type = response.headers.get('content-type', '')
        if content_type.startswith('application/json'):
            return response.json()

        text = response.text.strip()
        lines = [line for line in text.splitlines() if not line.lstrip().startswith('<')]
        if not lines:
            raise ValueError('No JSON payload found in response body')
        return json.loads(lines[-1])

    def get_json(self, params: dict[str, Any]) -> Any:
        cache_file = self._cache_path(params)
        if cache_file.exists():
            logger.info('cache hit: %s %s', params.get('api'), cache_file.name)
            with cache_file.open('r', encoding='utf-8') as f:
                return json.load(f)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                logger.info('fetching: %s params=%s attempt=%s/%s', params.get('api'), params, attempt + 1, self.max_retries)
                response = self.session.get(
                    f'{BASE_URL}/results-api.php',
                    params=params,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = self._decode_response(response)

                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with cache_file.open('w', encoding='utf-8') as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                logger.info('saved: %s', cache_file)
                time.sleep(self.sleep_seconds)
                return payload
            except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                wait = min(60.0, self.sleep_seconds * (2 ** attempt))
                logger.warning('request failed: %s params=%s retry_in=%.1fs', exc, params, wait)
                time.sleep(wait)

        raise RuntimeError(f'Failed to fetch params={params}') from last_error

    def scrape_all(self) -> dict[str, Any]:
        logger.info('start scrape_all')
        index = self.get_json({'api': 'index'})

        all_data: dict[str, Any] = {'index': index, 'leagues': {}, 'events': {}}
        seasons = index.get('seasons', [])
        logger.info('seasons discovered: %s', len(seasons))
        for season_idx, season in enumerate(seasons, start=1):
            leagues = season.get('leagues', [])
            logger.info('season %s/%s: %s leagues', season_idx, len(seasons), len(leagues))
            for league_idx, league in enumerate(leagues, start=1):
                league_id = league.get('id')
                if league_id is None:
                    continue

                logger.info('league %s/%s id=%s', league_idx, len(leagues), league_id)
                season_results = self.get_json(
                    {'api': 'season_leagues_results', 'league': league_id}
                )
                all_data['leagues'][str(league_id)] = season_results

                events = season_results.get('events', [])
                logger.info('league id=%s events: %s', league_id, len(events))
                for event_idx, event in enumerate(events, start=1):
                    event_id = event.get('event_id') or event.get('id')
                    if event_id is None:
                        continue

                    logger.info('event %s/%s id=%s', event_idx, len(events), event_id)
                    event_results = self.get_json(
                        {'api': 'event_results', 'event_id': event_id}
                    )
                    event_record = {
                        'event': event,
                        'event_results': event_results,
                        'full_results': [],
                    }

                    result_urls: set[str] = set()
                    stack = [event_results]
                    while stack:
                        current = stack.pop()
                        if isinstance(current, dict):
                            for key, value in current.items():
                                if key in {'result_url', 'full_results_url'} and isinstance(value, str):
                                    result_urls.add(value)
                                elif isinstance(value, (dict, list)):
                                    stack.append(value)
                        elif isinstance(current, list):
                            stack.extend(current)

                    logger.info('event id=%s result urls: %s', event_id, len(result_urls))
                    for detail_idx, result_url in enumerate(sorted(result_urls), start=1):
                        logger.info('full result %s/%s for event id=%s', detail_idx, len(result_urls), event_id)
                        detail = self.get_json(
                            {'api': 'event_full_results', 'result_url': result_url}
                        )
                        event_record['full_results'].append(
                            {
                                'result_url': result_url,
                                'payload': detail,
                            }
                        )

                    all_data['events'][str(event_id)] = event_record

        return all_data


def scrape_all_to_file(raw_dir: Path, out_file: Path, sleep_seconds: float, max_retries: int) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    scraper = IFSCScraper(raw_dir=raw_dir, sleep_seconds=sleep_seconds, max_retries=max_retries)
    payload = scraper.scrape_all()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
