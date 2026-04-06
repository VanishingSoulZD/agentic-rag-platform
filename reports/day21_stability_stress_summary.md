# Day 21 — 并发稳定性与压力测试（100 并发）

## 测试目标与阈值

- 并发：`100`
- 总请求数：`2000`
- 我设定的验收阈值：
  - 错误率 `< 2%`
  - P95 RT `<= 1500ms`

## 瓶颈定位与修复

### 基线版本（问题）

- 架构：无队列、无背压；in-flight 超过软上限后立即拒绝。
- 结果：高并发突发时大量请求直接失败（过载丢弃）。

### 修复版本（改进）

- 架构：`bounded queue + worker pool + backpressure`
  - 有界队列：削峰
  - worker 池：平滑处理
  - 入队超时：避免无限堆积
- 效果：在 100 并发下显著降低错误率并保持可控 P95。

## 实验数据（来自 `reports/day21_stability_stress_results.json`）

| Mode | Error Rate | P50(ms) | P95(ms) | Success QPS | CPU估计 | GPU |
|---|---:|---:|---:|---:|---:|---|
| baseline_no_backpressure | 98.80% | 80.43 | 102.95 | 211.54 | 18.73% | N/A |
| improved_queue_backpressure | 0.00% | 403.45 | 435.45 | 245.39 | 8.23% | N/A |

## 验收结论

- 改进后：
  - 错误率 `0.00% < 2%` ✅
  - P95 `435.45ms <= 1500ms` ✅
- 因此本次 **100 并发稳定性目标达成**。

## 运行命令

```bash
python scripts/day21_stability_stress_test.py --concurrency 100 --total-requests 2000 --p95-threshold-ms 1500
```
