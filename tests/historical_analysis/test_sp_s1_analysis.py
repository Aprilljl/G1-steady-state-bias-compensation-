"""Adapted from private source analysis tests; synthetic rows only."""
import json, math, sys, pytest
from historical.thesis_eval.analysis import sp_s1_analysis as s1
from historical.thesis_eval.analysis.sp_design import static_map


def row(t, survived): return {'command_vx':'1','survival_time_s':str(t),'survived':str(survived)}


def test_boundaries_and_actual_exposure():
    result=s1.speed_table([row(2,False),row(20,False),row(60,False),row(60.0200005,True)],60)[0]
    assert result['early']==1; assert result['exposure_s']==pytest.approx(134.0200005)
    assert result['early_segment']=={'deaths':1,'exposure_s':54.0,'rate':1/54}
    assert result['late_segment']=={'deaths':1,'exposure_s':80.0,'rate':1/80}
    assert result['conditional_survival']==pytest.approx(1/3)


def test_zero_events_has_upper_bound():
    r=s1.speed_table([row(60.0200005,True)],60)[0]
    assert r['hazard']==0 and r['lo']==0 and r['hi']>0; assert r['late_segment']['exposure_s']==40


def test_static_matching_and_survivors():
    def cell(payload,u,mean,survived=True): return dict(payload_kg=str(payload),command_vx=str(u),mean_vx=str(mean),survived=str(survived))
    a=static_map([cell(0,1,1),cell(0,1,9,False),cell(5,1.5,2)]); b=static_map([cell(5,1.5,2.2),cell(0,1,1.1)])
    assert a[0]['n']==2 and a[0]['n_survived']==1 and a[0]['g']==1
    match=s1.compare_static(a,b); assert (match['payload_kg'],match['u'])==(5,1.5); assert math.isclose(match['absolute_difference'],.2)
    with pytest.raises(ValueError,match='grids'): s1.compare_static(a,b[:1])


def test_rate_ratio_equal_and_known_clopper_pearson():
    seg={'deaths':1,'exposure_s':10}; r=s1.rate_ratio(seg,seg); q=1-math.sqrt(.975)
    assert r['estimate']==1; assert r['lo']==pytest.approx(q/(1-q)); assert r['hi']==pytest.approx((1-q)/q); assert r['lo']<1<r['hi']


def test_rate_ratio_zero_and_swap():
    a,b={'deaths':0,'exposure_s':10},{'deaths':3,'exposure_s':20}; x,y=s1.rate_ratio(a,b),s1.rate_ratio(b,a)
    assert x['estimate']==0 and x['lo']==0 and x['hi']>0; assert y['estimate'] is None and y['estimate_status']=='infinite'; assert y['hi'] is None and y['hi_status']=='infinite'; assert y['lo']==pytest.approx(1/x['hi'])
    both=s1.rate_ratio(a,a); assert both['estimate_status']=='indeterminate_zero_events' and both['lo']==0 and both['hi_status']=='infinite'; assert s1.rate_ratio(a,{'deaths':0,'exposure_s':0})['hi_status']=='no_exposure'; json.dumps([x,y,both],allow_nan=False)
    c,d={'deaths':4,'exposure_s':12},{'deaths':9,'exposure_s':15}; x,y=s1.rate_ratio(c,d),s1.rate_ratio(d,c)
    assert y['estimate']==pytest.approx(1/x['estimate']); assert y['lo']==pytest.approx(1/x['hi']); assert y['hi']==pytest.approx(1/x['lo']); assert 0<x['lo']<x['estimate']<x['hi']


def test_cli_output_path_guards(monkeypatch,tmp_path):
    output=tmp_path/'out'; args=['s1','--a1',str(output),'--a2','a2','--b1','b1','--b1g','b1g','--json',str(output),'--md',str(tmp_path/'other'),'--overwrite']; monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(ValueError,match='distinct'): s1.main()
    output.write_text('preserve',encoding='utf-8'); args[2]='a1'; args.remove('--overwrite')
    with pytest.raises(FileExistsError): s1.main()
    assert output.read_text(encoding='utf-8')=='preserve'
