from __future__ import annotations

import os
import time
import random
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Deque, List, Optional

from huggingface_hub import list_repo_files, hf_hub_download

REPO_ID = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
REPO_TYPE = "dataset"

LOCAL_DIR = "/inspire/hdd/global_user/chengdongzhou-240108390137/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"

FOLDERS = [
    "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
]

# ====== 你可以调的参数 ======
MAX_WORKERS = 2              # 并发线程数：1~4建议
API_RPS = 2.0                # 全局请求速率上限（每秒请求数），建议 1.5~2.5
WINDOW_SECONDS = 1.0         # 速率窗口（秒）
SUBMIT_BATCH = 50            # 每次往线程池提交多少任务，避免瞬时爆发
MAX_RETRIES = 50             # 单文件最大重试次数
# ===========================

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    print("WARNING: 未检测到 HF_TOKEN/HUGGINGFACE_HUB_TOKEN。强烈建议设置 token，否则更容易触发限流。")


class RateLimiter:
    """
    简单滑动窗口限速：确保在 WINDOW_SECONDS 内最多 API_RPS*WINDOW_SECONDS 次请求。
    """
    def __init__(self, rps: float, window_seconds: float = 1.0):
        self.capacity = max(1, int(rps * window_seconds))
        self.window = window_seconds
        self.lock = threading.Lock()
        self.ts: Deque[float] = deque()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.time()
                # 清掉窗口外的时间戳
                while self.ts and now - self.ts[0] > self.window:
                    self.ts.popleft()
                if len(self.ts) < self.capacity:
                    self.ts.append(now)
                    return
                # 需要等到最早一个请求出窗口
                sleep_for = self.window - (now - self.ts[0]) + 0.001
            time.sleep(max(0.01, sleep_for))


rate_limiter = RateLimiter(API_RPS, WINDOW_SECONDS)


def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return ("429" in msg) or ("Too Many Requests" in msg) or ("rate limit" in msg.lower())


def parse_retry_after_seconds(exc: Exception) -> Optional[float]:
    """
    best-effort：如果底层异常里包含 Retry-After（有些情况下会出现在响应信息里），就提取。
    不保证一定能取到，取不到就返回 None。
    """
    msg = str(exc)
    # 很多库不会把 header 打出来；这里只做保守匹配
    # 例如 "... retrying in 2.6s" 这类你自己的日志已经有等待时间了
    return None


def download_with_retry(filename: str) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 全局限速：每次“可能触发 Hub API / 下载”的动作前 acquire
            rate_limiter.acquire()

            hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=filename,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
                resume_download=True,
            )
            return True

        except Exception as exc:
            # 指数退避（带抖动），429 时更“慢”
            if is_rate_limit_error(exc):
                base = min(60.0, 2 ** min(attempt, 6))  # 2,4,8,16,32,64(封顶到60)
                jitter = random.uniform(0.0, 1.0)
                wait_time = base + jitter
            else:
                # 其他错误：稍微快一点，但也别太密
                wait_time = min(30.0, 1.5 * attempt) + random.uniform(0.0, 0.5)

            ra = parse_retry_after_seconds(exc)
            if ra is not None:
                wait_time = max(wait_time, ra)

            print(
                f"[{attempt}/{MAX_RETRIES}] "
                f"Download failed: {filename} ({exc}); retrying in {wait_time:.1f}s"
            )
            time.sleep(wait_time)

    print(f"Giving up after {MAX_RETRIES} retries: {filename}")
    return False


def iter_target_files(all_files: List[str]) -> List[str]:
    prefixes = tuple(folder + "/" for folder in FOLDERS)
    return [f for f in all_files if f.startswith(prefixes)]


def main() -> None:
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print("Listing all files in the repository (with token)...")
    # list_repo_files 本身也会走 Hub API，记得限速
    rate_limiter.acquire()
    all_files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE, token=HF_TOKEN)

    target_files = iter_target_files(all_files)
    print(f"Found {len(target_files)} matching files to download.\n")

    print(f"Starting download: workers={MAX_WORKERS}, api_rps={API_RPS}, submit_batch={SUBMIT_BATCH}\n")

    failed_files: List[str] = []
    idx_done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 分批提交，避免一次性创建海量 future 造成瞬时压力
        for batch_start in range(0, len(target_files), SUBMIT_BATCH):
            batch = target_files[batch_start: batch_start + SUBMIT_BATCH]
            futures = {executor.submit(download_with_retry, f): f for f in batch}

            for future in as_completed(futures):
                filename = futures[future]
                idx_done += 1
                try:
                    ok = future.result()
                    if ok:
                        print(f"[{idx_done}/{len(target_files)}] Downloaded: {filename}")
                    else:
                        failed_files.append(filename)
                except Exception as exc:
                    print(f"Unexpected error for {filename}: {exc}")
                    failed_files.append(filename)

            # 每批之间稍微喘口气，进一步降低“窗口峰值”
            time.sleep(0.5)

    print("\nAll download attempts finished.")
    if failed_files:
        print(f"{len(failed_files)} files failed to download:")
        for f in failed_files:
            print(f"  - {f}")
    else:
        print("All files downloaded successfully.")


if __name__ == "__main__":
    main()