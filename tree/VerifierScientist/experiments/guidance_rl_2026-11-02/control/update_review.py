"""Fill the single review document after the authorized optimizer finishes."""
import json
from pathlib import Path
from datetime import datetime,timezone

def update(out):
    out=Path(out);result=json.loads((out/'initial_optimization/result.json').read_text())
    selection=json.loads((out/'initial_optimization/selection.json').read_text())
    account=json.loads((out/'api_budget.json').read_text())
    accepted=result['decision']['accepted']
    rows=[]
    for name,m in selection['metrics'].items():
        fmt=lambda x:'无可用数据' if x is None else f'{x:.3f}'
        rows.append(f"| {name} | {fmt(m['macro_f1'])} | {fmt(m['pair_acc'])} |")
    body='\n'.join([
       '### 初始优化结果（自动填入实际产物）','',
       '更新时间：'+datetime.now(timezone.utc).isoformat(),'',
       '选择集候选：`'+result['winner']+'`；确认结论：**'+('接受候选' if accepted else '保留初始指南')+'**。这只是离线准入，不是 Router 训练已改善的证据。','',
       '| 候选 | 选择集 F1（较差重复） | pairAcc（较差重复） |','|---|---:|---:|',*rows,'',
       f"截至填入时，API 账本累计费用上界约 ${account['spent']:.3f}，另保留未知用量/在途预留约 ${sum(account['reserved'].values()):.3f}；硬上限仍为 $10。",'',
       '[完整选择/确认结果](../experiments/guidance_rl_2026-11-02/initial_optimization/result.json)、[最终指南版本](../experiments/guidance_rl_2026-11-02/initial_optimization/artifact.json)。',''])
    doc=out.parents[1]/'docs/05_Guidance优化与完整训练框架_Claude审核_2026-11-02.md'
    start='<!-- INITIAL_OPTIMIZATION_RESULT -->';end='<!-- END_INITIAL_OPTIMIZATION_RESULT -->'
    text=doc.read_text()
    if start not in text or end not in text:raise ValueError('Review document update markers missing; preserve document')
    before,tail=text.split(start,1);_,after=tail.split(end,1)
    doc.write_text(before+start+'\n'+body+end+after)

if __name__=='__main__':update(Path(__file__).resolve().parents[1])
