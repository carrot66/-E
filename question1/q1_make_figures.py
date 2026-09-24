from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
import numpy as np
import pandas as pd

required = {ord(char) for char in "问题补充样本时长词数特征维度对齐置信度视觉语音质量错误率复现流程"}
for family in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei"):
    candidates = [font for font in font_manager.fontManager.ttflist if font.name == family]
    if any(required.issubset(FT2Font(font.fname).get_charmap()) for font in candidates):
        mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
                             "axes.unicode_minus": False, "axes.axisbelow": True,
                             "svg.fonttype": "path", "pdf.fonttype": 42})
        break
else:
    raise RuntimeError("未找到包含问题一图表中文字的字体")

ROOT = Path(__file__).resolve().parents[1]
_DATA_CANDIDATES = (
    ROOT / "outputs" / "question1" / "问题1_全量特征结果",
    ROOT.parent / "q1_plot_data" / "问题1_全量特征结果",
    ROOT / "outputs" / "问题1_全量特征结果",
)
_ASR_CANDIDATES = (
    ROOT / "outputs" / "question1" / "问题1_ASR独立评估",
    ROOT.parent / "q1_plot_data" / "问题1_ASR独立评估",
    ROOT / "outputs" / "问题1_ASR独立评估",
)
DATA = next((path for path in _DATA_CANDIDATES if path.exists()), _DATA_CANDIDATES[0])
ASR = next((path for path in _ASR_CANDIDATES if path.exists()), _ASR_CANDIDATES[0])
OUT = ROOT / "outputs" / "question1" / "问题1_补充结果图"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#4472A8"
TEAL = "#4D8E87"
VIOLET = "#7B6AA8"
GREEN = "#6A9858"
RED = "#B86F69"
GOLD = "#B98930"
GREY = "#6B7680"
LIGHT = "#E8EEF2"
DARK = "#272727"


def style(size=11):
    plt.rcParams.update({
        "font.size": size,
        "axes.linewidth": 1.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#C9D3DA",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, name):
    fig.tight_layout(pad=1.2)
    base = OUT / name
    svg = Path(str(base) + ".svg")
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text(re.sub(r"[ \t]+(?=\r?$)", "", svg.read_text(encoding="utf-8"),
                          flags=re.MULTILINE), encoding="utf-8")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(base) + ".tif", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_label(ax, s):
    ax.text(-0.08, 1.04, {"a":"甲","b":"乙","c":"丙","d":"丁"}.get(s,s),
            transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")


def load():
    m = pd.read_csv(DATA / "sample_manifest.csv")
    s = pd.read_csv(DATA / "问题1_100条样本特征汇总.csv")
    a = pd.read_csv(ASR / "第一问_ASR逐样本结果.csv")
    q = json.loads((DATA / "quality_audit.json").read_text(encoding="utf-8"))
    sm = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    return m, s, a, q, sm


def fig01_coverage(m, q, sm):
    style(11)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6))
    ax = axs[0, 0]
    cats = ["样本", "特征文件", "逐词时间表"]
    vals = [len(m), int(q["feature_files"]), int(q["total_word_rows"])]
    ax.axis("off"); ax.set_title("全量覆盖与可追溯性")
    for i, (name, value, color) in enumerate(zip(cats, vals, [BLUE, TEAL, VIOLET])):
        x=.17+i*.33
        ax.text(x,.58,f"{value:,}",transform=ax.transAxes,ha="center",va="center",
                fontsize=23,fontweight="bold",color=color)
        ax.text(x,.37,name,transform=ax.transAxes,ha="center",va="center",fontsize=10)
    add_label(ax, "a")

    ax = axs[0, 1]
    names = ["文本", "语音", "视觉", "面部动作"]
    vals = [100, 98, 100, q["mean_face_blendshape_valid_rate"]*100]
    bars = ax.barh(names, vals, color=[BLUE, TEAL, VIOLET, GREEN])
    ax.set_xlim(0, 105); ax.set_xlabel("有效覆盖率 (%)"); ax.set_title("模态有效覆盖率")
    for b, v in zip(bars, vals): ax.text(v+1, b.get_y()+b.get_height()/2, f"{v:.1f}%", va="center", fontsize=9)
    add_label(ax, "b")

    ax = axs[1, 0]
    methods = ["CTC强制对齐", "比例回退"]
    counts = [int((m.ctc_alignment == "wav2vec2_ctc_forced_alignment").sum()), int((m.ctc_alignment != "wav2vec2_ctc_forced_alignment").sum())]
    bars = ax.bar(methods, counts, color=[TEAL, GOLD], width=.55)
    ax.set_ylabel("样本数"); ax.set_title("时间对齐方法构成")
    for b, v in zip(bars, counts): ax.text(b.get_x()+b.get_width()/2, v+1, str(v), ha="center")
    ax.set_ylim(0, max(counts)+12); add_label(ax, "c")

    ax = axs[1, 1]
    labels = ["全量样本", "静音样本", "CTC置信度覆盖", "人脸检测词覆盖"]
    vals = [100, int(q["audio_missing_samples"]), q["ctc_word_confidence_coverage"]*100, q["mean_face_detection_word_rate"]*100]
    ax.bar(labels, vals, color=[BLUE, RED, VIOLET, GREEN])
    ax.set_ylim(0, 105); ax.set_ylabel("数量或覆盖率 (%)"); ax.set_title("审计关键指标")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(vals): ax.text(i, v+2, f"{v:.1f}" if isinstance(v, float) else str(v), ha="center", fontsize=9)
    add_label(ax, "d")
    fig.suptitle("问题1补充图1｜100条样本的覆盖、对齐与质量门禁", fontsize=15, fontweight="bold")
    save(fig, "q1_fig01_全量覆盖与审计")


def fig02_duration(m):
    style(11)
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 4.3))
    d = m.duration_video_sec.to_numpy(float); w = m.word_count.to_numpy(float)
    ax = axs[0]; ax.hist(d, bins=14, color=BLUE, alpha=.88, edgecolor="none"); ax.axvline(np.median(d), color=RED, ls="--", label=f"中位数 {np.median(d):.2f}s"); ax.set_xlabel("有效视频时长 (s)"); ax.set_ylabel("样本数"); ax.legend(fontsize=9); ax.set_title("时长分布"); ax.grid(False); add_label(ax, "a")
    ax = axs[1]; ax.hist(w, bins=np.arange(w.min(), w.max()+2, 2), color=TEAL, alpha=.88, edgecolor="none"); ax.axvline(np.median(w), color=RED, ls="--", label=f"中位数 {np.median(w):.0f}词"); ax.set_xlabel("逐词特征长度"); ax.set_ylabel("样本数"); ax.legend(fontsize=9); ax.set_title("词数分布"); ax.grid(False); add_label(ax, "b")
    ax = axs[2]; c = np.where(m.ctc_alignment.str.contains("forced"), TEAL, GOLD); ax.scatter(d, w, c=c, s=28, alpha=.85, edgecolors="#FFFFFF", linewidth=.4); ax.set_xlabel("有效视频时长 (s)"); ax.set_ylabel("词数"); ax.set_title("时长与序列长度对应"); ax.grid(True, color="#C9D3DA", linewidth=.7); add_label(ax, "c")
    fig.suptitle("问题1补充图2｜时长和逐词序列长度的全量统计", fontsize=15, fontweight="bold")
    save(fig, "q1_fig02_时长与词数分布")


def fig03_dimensions(s):
    style(11)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axs[0]
    dims = [768, 74, 768, 52, 49, 512]
    names = ["文本\nBERT", "语音\n声学", "语音\nwav2vec", "视觉\nBlendshape", "视觉\nLBP", "视觉\nResNet18"]
    colors = [BLUE, TEAL, TEAL, GREEN, GREEN, VIOLET]
    bars=ax.bar(names, dims, color=colors, width=.65)
    ax.set_ylabel("每词特征维度"); ax.set_title("三模态特征维度")
    for b,v in zip(bars,dims): ax.text(b.get_x()+b.get_width()/2,v+max(dims)*.025,str(v),ha="center",fontsize=9)
    ax.set_ylim(0, 900); ax.tick_params(axis="x",labelsize=8,rotation=18); ax.grid(False); add_label(ax,"a")
    ax = axs[1]
    df = s.copy(); df["align"] = np.where(df["对齐方法"].str.contains("forced"), "CTC强制对齐", "比例回退")
    data=[df.loc[df["align"]==x,"CTC置信度"].dropna().to_numpy() for x in ["CTC强制对齐","比例回退"]]
    ax.boxplot(data, tick_labels=[f"强制对齐\n{len(data[0])}条",f"低置信度回退\n{len(data[1])}条"], patch_artist=True, boxprops=dict(facecolor=LIGHT), medianprops=dict(color=RED,linewidth=2)); ax.axhline(-3,color=GOLD,ls="--",label="回退阈值 -3"); ax.set_ylabel("平均字符对数置信度"); ax.set_title("对齐置信度分层"); ax.legend(fontsize=9); ax.grid(False); add_label(ax,"b")
    ax.text(.98,.05,"另有2条静音样本无CTC分数",transform=ax.transAxes,ha="right",va="bottom",fontsize=8,color=GREY)
    fig.suptitle("问题1补充图3｜特征维度固定性与对齐置信度", fontsize=15, fontweight="bold")
    save(fig,"q1_fig03_维度与对齐置信度")


def fig04_heatmap(m):
    style(9)
    order = m.sort_values(["ctc_alignment","ctc_log_confidence"]).reset_index(drop=True)
    mat=np.column_stack([order.face_detection_rate, order.face_blendshape_detection_rate, 1-order.vision_nearest_frame_fallback_rate, order.ctc_log_confidence.clip(-8,0)/8+1])
    fig, ax = plt.subplots(figsize=(12,7.0)); im=ax.imshow(mat.T,aspect="auto",cmap="YlGnBu",vmin=0,vmax=1)
    ax.set_yticks(range(4),["人脸检测率","Blendshape有效率","视觉原帧命中率","CTC置信度归一化"]); ax.set_xlabel("按对齐方法与置信度排序的样本序号"); ax.set_title("逐样本质量门禁热图")
    ax.set_xticks(np.arange(0,len(order),10), [str(i+1) for i in range(0,len(order),10)]); ax.grid(False); fig.colorbar(im,ax=ax,label="比例 / 归一化质量")
    ax.axvline((order.ctc_alignment!="wav2vec2_ctc_forced_alignment").sum()-.5,color=RED,ls="--",lw=1.3,label="对齐回退分界"); ax.legend(loc="upper right",fontsize=9); add_label(ax,"a")
    fig.suptitle("问题1补充图4｜100条样本的跨模态质量分布", fontsize=15, fontweight="bold")
    save(fig,"q1_fig04_逐样本质量热图")


def fig05_asr(a):
    style(10)
    fig, axs=plt.subplots(1,2,figsize=(12,4.8))
    groups=[]; vals=[]
    labels={"proportional_fallback_low_ctc_confidence":"低置信度回退",
            "proportional_fallback_silent_audio":"静音回退",
            "wav2vec2_ctc_forced_alignment":"CTC强制对齐"}
    for key, sub in a.groupby("ctc_alignment"):
        groups.append(f"{labels[key]}\n{sub.shape[0]}条")
        vals.append([sub.sample_WER.mean(),sub.sample_CER.mean()])
    arr=np.asarray(vals); x=np.arange(len(groups)); w=.34; ax=axs[0]; ax.bar(x-w/2,arr[:,0],w,label="词错误率",color=BLUE); ax.bar(x+w/2,arr[:,1],w,label="字错误率",color=VIOLET); ax.set_xticks(x,groups); ax.set_ylabel("错误率"); ax.set_ylim(0,max(1.25,arr.max()*1.12)); ax.set_title("按对齐方法的识别错误率"); ax.legend(); ax.grid(False); add_label(ax,"a")
    ax=axs[1]; sub=a.sort_values("sample_WER").reset_index(drop=True); x=np.arange(len(sub)); ax.plot(x,sub.sample_WER,color=BLUE,lw=1.3,label="词错误率"); ax.plot(x,sub.sample_CER,color=VIOLET,lw=1.3,label="字错误率"); ax.axhline(a.sample_WER.mean(),color=BLUE,ls="--",alpha=.6); ax.axhline(a.sample_CER.mean(),color=VIOLET,ls="--",alpha=.6); ax.set_xlabel("按词错误率排序的样本序号"); ax.set_ylabel("错误率"); ax.set_ylim(0,max(sub.sample_WER.max(),sub.sample_CER.max())*1.08); ax.set_title("逐样本识别错误率排序"); ax.legend(); ax.grid(True,color="#C9D3DA",linewidth=.7); add_label(ax,"b")
    fig.suptitle("问题1补充图5｜ASR质量与对齐策略诊断",fontsize=15,fontweight="bold"); save(fig,"q1_fig05_ASR质量诊断")


def fig06_typical():
    f=next((DATA/"features").glob("-3g5yACwYnA__13.npz")); z=np.load(f,allow_pickle=True); words=z["words"].astype(str); t=z["word_times"]; x=np.arange(len(words));
    text=np.linalg.norm(z["text"],axis=1); audio=np.linalg.norm(z["audio_wav2vec"],axis=1); vis=np.linalg.norm(z["vision_resnet18"],axis=1); conf=z["ctc_char_log_confidence"]
    fig,axs=plt.subplots(3,1,figsize=(12,7.8),sharex=True,gridspec_kw={"height_ratios":[1,1,1.25]})
    ax=axs[0]; ax.plot(x,text,"o-",color=BLUE,label="文本 BERT"); ax.plot(x,audio,"o-",color=TEAL,label="语音 wav2vec2"); ax.plot(x,vis,"o-",color=VIOLET,label="视觉 ResNet18"); ax.set_ylabel("每词特征L2范数"); ax.set_title("典型样本：三模态逐词特征序列"); ax.legend(ncol=3,fontsize=9); add_label(ax,"a")
    ax=axs[1]; ax.bar(x,conf,color=np.where(conf<-3,RED,TEAL),width=.72); ax.axhline(-3,color=GOLD,ls="--",label="回退阈值"); ax.set_ylabel("CTC字符置信度"); ax.legend(fontsize=9); add_label(ax,"b")
    ax=axs[2]; ax.imshow(np.vstack([z["text_valid"],z["audio_valid"],z["vision_valid"],z["face_blendshape_valid"]]).astype(int),aspect="auto",cmap="Greens",vmin=0,vmax=1,interpolation="nearest"); ax.set_yticks(range(4),["文本有效","语音有效","视觉有效","面部动作有效"]); ax.set_xticks(x,words,rotation=35,ha="right"); ax.set_xlabel("词序列（对应同一时间区间）"); ax.grid(False); add_label(ax,"c")
    fig.suptitle("问题1补充图6｜典型样本 -3g5yACwYnA__13 的特征级对应关系",fontsize=15,fontweight="bold"); save(fig,"q1_fig06_典型样本特征序列")


def fig07_quality_rank(m):
    d=m.copy(); d["quality_score"]=(d.face_detection_rate.fillna(0)+d.face_blendshape_detection_rate.fillna(0)+(1-d.vision_nearest_frame_fallback_rate.fillna(0))+(d.ctc_log_confidence.clip(-8,0)/8+1))/4; d=d.sort_values("quality_score").reset_index(drop=True); fig,ax=plt.subplots(figsize=(12,4.8)); x=np.arange(len(d)); colors=np.where(d.quality_score<.5,RED,np.where(d.quality_score<.75,GOLD,GREEN)); ax.scatter(x,d.quality_score,c=colors,s=24); ax.axhline(.75,color=GREEN,ls="--",lw=1,label="0.75参考线"); ax.axhline(.5,color=RED,ls="--",lw=1,label="0.50参考线"); ax.set_xlabel("按综合质量分数排序的样本序号"); ax.set_ylabel("综合质量分数"); ax.set_ylim(0,1.05); ax.set_title("逐样本综合质量排序（用于人工复核优先级）"); ax.legend(fontsize=9); add_label(ax,"a");
    low=d.head(5)
    priority="优先复核样本：\n"+"\n".join(f"{r.sample_id}  {r.quality_score:.2f}" for _,r in low.iterrows())
    ax.text(.98,.05,priority,transform=ax.transAxes,ha="right",va="bottom",fontsize=8,
            bbox=dict(boxstyle="round,pad=.4",fc="white",ec="#CCD4DA"))
    ax.grid(False); fig.suptitle("问题1补充图7｜异常样本筛查与人工复核优先级",fontsize=15,fontweight="bold"); save(fig,"q1_fig07_样本质量排序")


def fig08_pipeline(q, sm):
    style(11); fig,ax=plt.subplots(figsize=(13,5.4)); ax.axis("off");
    boxes=[("原始视频/音频/转写", "100条样本"), ("文本编码", "BERT 768-D"), ("语音编码", "wav2vec2 + 74-D"), ("CTC逐词对齐", "1,934个词区间"), ("视觉编码", "Blendshape + LBP + ResNet18"), ("统一NPZ输出", "100/100可追溯")]
    xs=np.linspace(.08,.92,len(boxes)); y=.6
    for i,(title,sub) in enumerate(boxes):
        fc=["#E8EEF2","#DDEBFA","#DFF2F0","#FFF0D0","#E8F2DF","#E5DEF3"][i]; ax.text(xs[i],y,title,ha="center",va="center",fontsize=11,fontweight="bold",bbox=dict(boxstyle="round,pad=.75",fc=fc,ec="#4D4D4D",lw=1.1)); ax.text(xs[i],y-.14,sub,ha="center",va="center",fontsize=9,color="#4D4D4D")
        if i<len(boxes)-1: ax.annotate("",xy=(xs[i+1]-.075,y),xytext=(xs[i]+.075,y),arrowprops=dict(arrowstyle="->",lw=1.5,color="#4D4D4D"))
    metrics=[("样本覆盖",f"{sm['successful_or_existing']}/{sm['expected_samples']}"),("CTC置信度覆盖",f"{q['ctc_word_confidence_coverage']*100:.1f}%"),("人脸检测词覆盖",f"{q['mean_face_detection_word_rate']*100:.1f}%"),("Blendshape有效率",f"{q['mean_face_blendshape_valid_rate']*100:.1f}%"),("回退样本",f"{q['ctc_fallback_samples']}")]
    for i,(k,v) in enumerate(metrics): ax.text(.12+i*.19,.19,k+"\n"+v,ha="center",va="center",fontsize=10,bbox=dict(boxstyle="round,pad=.45",fc="white",ec="#B9C3CA"))
    ax.text(.5,.92,"问题1补充图8｜从原始素材到标准化逐词特征的可复现流程",ha="center",fontsize=15,fontweight="bold")
    ax.text(.5,.05,"时间主键：词区间 [t_start,t_end]；语音取区间内10 ms帧，视觉取区间内10 fps帧；无命中时记录最近帧回退。",ha="center",fontsize=9,color="#4D4D4D")
    save(fig,"q1_fig08_可复现处理流程")


def main():
    m,s,a,q,sm=load(); fig01_coverage(m,q,sm); fig02_duration(m); fig03_dimensions(s); fig04_heatmap(m); fig05_asr(a); fig06_typical(); fig07_quality_rank(m); fig08_pipeline(q,sm)
    manifest={"figure_count":len(list(OUT.glob("q1_fig*.png"))),"source_samples":int(len(m)),"source_word_rows":int(q["total_word_rows"]),"output_dir":"outputs/question1/问题1_补充结果图","repository_formats":["svg","pdf","png"],"local_only_format":"tif","note":"补充图避开已有的词语-语音-视频帧复合图，专注全量统计、质量审计、特征序列和复现流程。"}
    if (OUT/"q1_fig09_语音特征处理前后对比.png").is_file():
        manifest["audio_comparison_figure"]="q1_fig09_语音特征处理前后对比"
    (OUT/"figure_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
