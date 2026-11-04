"""Exactly 12 PowerPoint slides: 3 each (RF, XGBoost, LightGBM, CatBoost)."""
import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/"outputs"/"metrics.json").read_text(encoding="utf-8"))
metrics={r["model"]:r for r in p["metrics"]}
spec=[
 ("Random Forest","Ансамбль деревьев голосует за класс: ушёл или остался.",
  "100 деревьев • n_jobs=1","Простой baseline, но возможен сильный overfitting."),
 ("XGBoost","Деревья обучаются последовательно и исправляют ошибки предыдущих.",
  "300 деревьев • lr=0.05 • depth=4","Хорошо работает на табличных данных; важна настройка."),
 ("LightGBM","Градиентный бустинг с эффективным построением деревьев.",
  "300 деревьев • lr=0.05 • depth=4","Может быть быстрым, особенно на больших данных."),
 ("CatBoost","Градиентный бустинг с прямой обработкой категорий.",
  "300 деревьев • lr=0.05 • depth=4","region и tariff переданы без one-hot: удобно для категорий."),
]
NAVY=RGBColor(10,31,68)
CARD=RGBColor(21,43,77)
WHITE=RGBColor(249,252,255)
GRAY=RGBColor(170,194,218)
CYAN=RGBColor(0,191,255)
GREEN=RGBColor(116,227,177)

prs=Presentation()
prs.slide_width=Inches(13.333)
prs.slide_height=Inches(7.5)
layout=prs.slide_layouts[6]

def text(slide,t,x,y,w,h,sz=20,col=WHITE,bold=False):
    tb=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    f=tb.text_frame
    f.clear()
    f.word_wrap=True
    f.margin_left=Inches(.05)
    f.margin_top=Inches(.02)
    for i,line in enumerate(t.split("\n")):
        q=f.paragraphs[0] if not i else f.add_paragraph()
        q.text=line
        q.font.name="Aptos"
        q.font.size=Pt(sz)
        q.font.bold=bold
        q.font.color.rgb=col
    return tb

def panel(s,x,y,w,h):
    sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb=CARD
    sh.line.fill.background()

def base(n,name,title):
    s=prs.slides.add_slide(layout)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb=NAVY
    text(s,"ML-P-8  /  "+name.upper(),.75,.5,11,.4,15,CYAN,True)
    text(s,title,.75,1.12,12,.85,31,WHITE,True)
    text(s,"3000 абонентов  •  train 2400 / test 600  •  random_state=42",.75,6.93,11,.28,11,GRAY)
    text(s,f"{n:02d} / 12",11.64,6.93,1,.28,11,GRAY)
    return s

for idx,(name,idea,settings,description) in enumerate(spec):
    m=metrics[name]
    n=idx*3
    s=base(n+1,name,"Принцип работы и настройки")
    panel(s,.75,2.16,7.13,3.7)
    panel(s,8.12,2.16,4.45,3.7)
    text(s,"Идея алгоритма",1.03,2.56,6.3,.48,23,CYAN,True)
    text(s,idea,1.03,3.36,6.25,1.4,23)
    text(s,"Конфигурация",8.41,2.56,3.8,.48,23,CYAN,True)
    text(s,settings+"\n\nОдин train/test split.",8.41,3.37,3.78,1.9,18)
    s.notes_slide.notes_text_frame.text=(name+": "+idea+" "+settings+" Данные разделены одинаково для всех алгоритмов.")

    s=base(n+2,name,"Качество и риск переобучения")
    for j,(label,value) in enumerate([("F1 TRAIN",m["f1_train"]),("F1 TEST",m["f1_test"]),("ROC-AUC TEST",m["roc_auc_test"])]):
        x=.75+j*4.09
        panel(s,x,2.18,3.85,2.43)
        text(s,label,x+.23,2.51,3.5,.38,16,GRAY)
        text(s,f"{value:.3f}",x+.23,3.22,3.4,.77,39,CYAN if j<2 else GREEN,True)
    panel(s,.75,4.94,12.02,.95)
    text(s,f"Разрыв F1 train/test: {m['f1_gap']:.3f}      •      Медиана обучения: {m['train_seconds_median']:.3f} с",
         1.03,5.21,11.45,.48,21,WHITE,True)
    s.notes_slide.notes_text_frame.text=(f"{name}. F1 train {m['f1_train']:.3f}; F1 test {m['f1_test']:.3f};"
                                         f" AUC {m['roc_auc_test']:.3f}. Чем больше разрыв, тем выше риск переобучения.")

    s=base(n+3,name,"Практический вывод")
    panel(s,.75,2.15,12.0,1.84)
    text(s,"Сильные стороны и ограничения",1.04,2.48,11.5,.46,22,CYAN,True)
    text(s,description,1.04,3.18,11.2,.53,20,WHITE)
    panel(s,.75,4.34,12.0,1.35)
    text(s,f"Инференс 600 строк: {m['inference_seconds_median']:.5f} с    •    Размер: {m['model_size_kb']:.1f} КБ",
         1.04,4.68,11.4,.55,19,WHITE,True)
    if name=="CatBoost":
        text(s,"ИТОГ: лучший бустинг по ROC-AUC test — "+p["best_boost_auc"],.83,6.07,11.8,.62,23,GREEN,True)
    else:
        text(s,"Полное сравнение и рекомендация — в README репозитория.",.83,6.07,11.8,.62,21,GREEN,True)
    s.notes_slide.notes_text_frame.text=(f"{name}. {description} Время инференса {m['inference_seconds_median']:.5f} сек"
                                         f" для всех тестовых строк. Размер модели {m['model_size_kb']:.1f} КБ."
                                         + (" Лучший бустинг по AUC: "+p["best_boost_auc"]+"." if name=="CatBoost" else ""))

dest=ROOT/"presentation"/"ML_P8_Comparison.pptx"
dest.parent.mkdir(exist_ok=True)
prs.save(dest)
print("PowerPoint slides:",len(prs.slides),"path:",dest)
