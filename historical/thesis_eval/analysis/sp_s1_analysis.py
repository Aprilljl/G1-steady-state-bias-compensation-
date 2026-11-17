"""S1 descriptive analysis and conditional exact segmented rate-ratio intervals."""
import argparse
import json
import math
from pathlib import Path

from scipy.stats import beta

if __package__:
    from .sp_design import hazard_table, static_map, read_csv, truthy
else:
    from sp_design import hazard_table, static_map, read_csv, truthy


def segment(rows, start, end):
    """Exposure in (start,end], counting deaths at end but not at start."""
    exposure = sum(max(0.0, min(float(r['survival_time_s']), end) - start) for r in rows)
    deaths = sum(not truthy(r['survived']) and start < float(r['survival_time_s']) <= end for r in rows)
    return {'deaths': deaths, 'exposure_s': exposure, 'rate': deaths / exposure if exposure else None}


def rate_ratio(a, b):
    """Early/late rate ratio; conditional Poisson exact CI, fixed observed exposures."""
    d0, d1 = a['deaths'], b['deaths']
    e0, e1 = a['exposure_s'], b['exposure_s']
    if e0 <= 0 or e1 <= 0:
        return {'estimate': None, 'estimate_status': 'no_exposure',
                'lo': None, 'lo_status': 'no_exposure', 'hi': None, 'hi_status': 'no_exposure'}
    n = d0 + d1
    if not n:
        return {'estimate': None, 'estimate_status': 'indeterminate_zero_events',
                'lo': 0.0, 'lo_status': 'finite', 'hi': None, 'hi_status': 'infinite'}
    qlo = beta.ppf(.025, d0, d1 + 1) if d0 else 0.0
    qhi = beta.isf(.025, d0 + 1, d1) if d1 else 1.0
    def transform(q):
        return q / (1 - q) * e1 / e0
    return {'estimate': (d0 / e0) / (d1 / e1) if d1 else None,
            'estimate_status': 'finite' if d1 else 'infinite',
            'lo': transform(qlo), 'lo_status': 'finite',
            'hi': transform(qhi) if d1 else None, 'hi_status': 'finite' if d1 else 'infinite'}


def speed_table(rows, nominal):
    result = []
    for h in hazard_table(rows):
        group = [r for r in rows if float(r['command_vx']) == h['command_vx']]
        eligible = [r for r in group if float(r['survival_time_s']) > 2]
        survivors = [r for r in group if truthy(r['survived'])]
        result.append({**h, 'survived': len(survivors), 'full_survival': len(survivors) / len(group),
                       'conditional_survival': len(survivors) / len(eligible) if eligible else None,
                       'survivor_time_min_s': min((float(r['survival_time_s']) for r in survivors), default=None),
                       'survivor_time_max_s': max((float(r['survival_time_s']) for r in survivors), default=None),
                       'early_segment': segment(group, 2, 20), 'late_segment': segment(group, 20, nominal)})
        result[-1]['segment_rate_ratio'] = rate_ratio(result[-1]['early_segment'], result[-1]['late_segment'])
    return result


def compare_static(a, b):
    def index(rows):
        result = {(r['payload_kg'], r['u']): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate static cell')
        return result
    left, right = index(a), index(b)
    if left.keys() != right.keys():
        raise ValueError('static grids do not match')
    comparisons = []
    for key in sorted(left):
        x, y = left[key], right[key]
        if key[1] <= 1.5 and math.isfinite(x['g']) and math.isfinite(y['g']):
            comparisons.append({'payload_kg': key[0], 'u': key[1], 'mujoco_g': x['g'], 'isaac_g': y['g'], 'absolute_difference': abs(x['g'] - y['g'])})
    return max(comparisons, key=lambda x: x['absolute_difference']) if comparisons else None


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def report(data):
    lines = ['# S1 描述性基础表', '', '原记录总率使用实际记录时长，2 s 前（含 2 s）退出不进入暴露；95% CI 为 Garwood 区间。零事件的正上限不代表零风险。全回合存活率为记录的 survived/n，条件存活率为 survived/时长 > 2 s 的数量。', '', '名义窗分段暴露截到名义终点，事件按 (2,20]、(20,终点] 归属；同一回合可贡献两段暴露，因此分段之和可能不同于未截断的原记录总率。率比为前段/后段；95% CI 按条件 Poisson 精确法，以二项 Clopper–Pearson 区间和暴露比变换，条件于观察到的暴露；分段暴露与事件来自同一生存回合，独立 Poisson 假设是近似，区间不证明恒定失效率。各速度为描述性呈现，不作多重检验或新假设声明。', '']
    for name, nominal in [('SP-A1', 120), ('SP-A2', 60)]:
        times = sorted({v for r in data[name] for v in (r['survivor_time_min_s'], r['survivor_time_max_s']) if v is not None})
        lines += [f"{name} 名义终点 {nominal} s；存活记录时长及相对名义终点差值：" + ', '.join(f'{v:.10g} s ({v - nominal:+.10g} s)' for v in times) + '。', '']
    for name, nominal in [('SP-A1', 120), ('SP-A2', 60)]:
        lines += [f'## {name}（名义 {nominal} s）', '', '|速度|n|≤2 s|存活|全回合存活|条件存活|原记录 D/E|原记录率 [95% CI]|存活记录时长范围 s|[2,20] D/E (率)|(20,名义终点] D/E (率)|', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in data[name]:
            def f(x): return 'NA' if x is None else f'{x:.8g}'
            a, b = r['early_segment'], r['late_segment']
            lines.append(f"|{r['command_vx']:g}|{r['n']}|{r['early']}|{r['survived']}|{f(r['full_survival'])}|{f(r['conditional_survival'])}|{r['deaths']}/{f(r['exposure_s'])}|{f(r['hazard'])} [{f(r['lo'])}, {f(r['hi'])}]|{f(r['survivor_time_min_s'])}–{f(r['survivor_time_max_s'])}|{a['deaths']}/{f(a['exposure_s'])} ({f(a['rate'])})|{b['deaths']}/{f(b['exposure_s'])} ({f(b['rate'])})|")
        lines += ['', '|速度|前段/后段率比|95% CI|', '|---:|---:|---:|']
        for r in data[name]:
            rr = r['segment_rate_ratio']
            def fmt(key):
                status = rr[key + '_status']
                return f"{rr[key]:.8g}" if status == 'finite' else {'infinite': '∞', 'no_exposure': '无暴露', 'indeterminate_zero_events': '未定（双零事件）'}[status]
            lines.append(f"|{r['command_vx']:g}|{fmt('estimate')}|[{fmt('lo')}, {fmt('hi')}]|")
        lines.append('')
    pa1 = next(r for r in data['SP-A1'] if r['command_vx'] == 2.5)
    ratio = pa1['segment_rate_ratio']
    lines += [f"PA-1：MuJoCo 2.5 m/s 条件存活 {pa1['survived']}/{pa1['n'] - pa1['early']} = {pa1['conditional_survival']:.8g}，落在原预测 0.48 [0.18, 0.79] 内；分段率比 {ratio['estimate']:.8g} [{ratio['lo']:.8g}, {ratio['hi']:.8g}] {'包含' if ratio['lo'] <= 1 <= ratio['hi'] else '不包含'} 1。此观察与登记预测一致，不证明模型成立；其他速度亦须查看上表，例如 2.4 m/s。", '']
    for name in ('SP-B1', 'SP-B1g'):
        lines += [f'## {name} 静态图（仅存活回合均值 g）', '', '|负载 kg|速度|n|存活|g|', '|---:|---:|---:|---:|---:|']
        for r in data[name]:
            g = 'NA' if r['g'] is None else f"{r['g']:.8g}"
            lines.append(f"|{r['payload_kg']:g}|{r['u']:g}|{r['n']}|{r['n_survived']}|{g}|")
        lines.append('')
    c = data['cross_engine_max']
    lines += [f"u≤1.5 的匹配工况最大绝对差：{c['absolute_difference']:.9g} m/s（负载 {c['payload_kg']:g} kg，u={c['u']:g} m/s；MuJoCo {c['mujoco_g']:.9g}，Isaac {c['isaac_g']:.9g}）；登记阈值 0.06 m/s。", '', 'PA-4 反例（MuJoCo，u≤2.0）：']
    for r in data['pa4_counterexamples']:
        lines.append(f"- {r['payload_kg']:g} kg、{r['u']:g} m/s：{r['n_survived']}/{r['n']} 存活。")
    lines += ['', '待办：Weibull、图和其他检验；未拒绝不等于模型成立。', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('a1', 'a2', 'b1', 'b1g', 'json', 'md'):
        p.add_argument('--' + arg, required=True)
    p.add_argument('--overwrite', action='store_true', help='replace existing report outputs')
    args = p.parse_args()
    targets = [Path(args.json), Path(args.md)]
    if len({p.resolve() for p in targets}) != 2 or any(p.resolve() in {Path(getattr(args, k)).resolve() for k in ('a1', 'a2', 'b1', 'b1g')} for p in targets):
        raise ValueError('output paths must be distinct from each other and inputs')
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError('refusing to overwrite output')
    data = {'SP-A1': speed_table(read_csv(args.a1), 120), 'SP-A2': speed_table(read_csv(args.a2), 60),
            'SP-B1': static_map(read_csv(args.b1)), 'SP-B1g': static_map(read_csv(args.b1g))}
    data['cross_engine_max'] = compare_static(data['SP-B1'], data['SP-B1g'])
    data['pa4_counterexamples'] = [r for r in data['SP-B1'] if r['u'] <= 2 and r['n_survived'] < r['n']]
    data['inputs'] = {k: str(Path(getattr(args, k)).resolve()) for k in ('a1', 'a2', 'b1', 'b1g')}
    data = clean(data)
    text = report(data)
    for path, content in zip(targets, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', text)):
        with path.open('w' if args.overwrite else 'x', encoding='utf-8', newline='\n') as f:
            f.write(content)
    print(f'wrote {targets[0]} and {targets[1]}')


if __name__ == '__main__':
    main()
