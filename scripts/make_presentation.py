"""Build 7 slide PowerPoint: 3 Random Forest, 3 XGBoost, 1 final."""
from pathlib import Path
import json
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT=Path(__file__).resolve().parents[1]
dest=ROOT/"presentation"/"ML_P8_Comparison.pptx"
obj=json.loads((ROOT/"outputs"/"metrics.json").read_text(encoding="utf-8"))
rf=next(row for row in obj["metrics"] if row["model"]=="Random Forest")
xgb=next(row for row in obj["metrics"] if row["model"]=="XGBoost")
NAVY=RGBColor(10,31,68)
DARK=RGBColor(16,28,48)
BLUE=RGBColor(0,191,255)
WHITE=RGBColor(255,255,255)
GRAY=RGBColor(177,195,215)
GREEN=RGBColor(101,230,175)
prs=Presentation()
prs.slide_width=Inches(13.333)
prs.slide_height=Inches(7.5)

def label(slide,msg,x,y,w,h,sz=20,col=WHITE,bold=False):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=box.text_frame
    tf.word_wrap=True
    for i,line in enumerate(msg.split("\n")):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text=line
        p.font.name="Aptos"
        p.font.size=Pt(sz)
        p.font.bold=bold
        p.font.color.rgb=col
    return box

def panel(slide,x,y,w,h):
    sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb=DARK
    sh.line.fill.background()

def slide(k,section,title):
    s=prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb=NAVY
    label(s,section.upper(),.85,.55,11,.35,14,BLUE,True)
    label(s,title,.85,1.05,11.5,.8,31,WHITE,True)
    label(s,"Результаты рассчитаны по CSV | random_state=42",.85,7.02,10,.22,10,GRAY)
    label(s,f"{k:02d} / 07",11.77,7.02,1,.22,10,GRAY)
    return s

def addcards(s,entries):
    for i,(heading,body) in enumerate(entries):
        x=.85+i*6.03
        panel(s,x,2.2,5.78,3.72)
        label(s,heading,x+.35,2.52,5.12,.5,23,BLUE,True)
        label(s,body,x+.35,3.35,5.08,2.3,20,WHITE)

def scorecards(s,m):
    for i,(name,key,color) in enumerate([
        ("F1 train","f1_train",GREEN),("F1 test","f1_test",BLUE),
        ("ROC-AUC test","roc_auc_test",GREEN)]):
        x=.85+i*4.04
        panel(s,x,2.18,3.82,3.12)
        label(s,name,x+.27,2.47,3.2,.46,18,GRAY)
        label(s,f"{m[key]:.3f}",x+.27,3.32,3.2,.9,40,color,True)
    label(s,f"Разрыв F1 train/test: {m['f1_gap']:.3f}",.85,5.72,11.7,.5,23,WHITE,True)

s=slide(1,"Random Forest • 1/3","Как работает случайный лес")
addcards(s,[("Принцип","Много деревьев учатся на случайных выборках. Решение формируется голосованием."),
            ("Наш эксперимент","100 деревьев, seed=42.\nОбщие данные и разбиение: 2400 train, 600 test.")])
s=slide(2,"Random Forest • 2/3","Метрики Random Forest")
scorecards(s,rf)
s=slide(3,"Random Forest • 3/3","Преимущества и ограничения")
addcards(s,[("Преимущества","Простой baseline.\nУлавливает нелинейные связи.\nНе требует масштабирования."),
            ("Риск","Большой разрыв train/test.\nМожет переобучиться.\nМожно ограничить глубину деревьев.")])
s=slide(4,"XGBoost • 1/3","Как работает градиентный бустинг")
addcards(s,[("Принцип","Деревья строятся последовательно. Каждое новое исправляет ошибки предыдущих."),
            ("Наш эксперимент","300 деревьев.\nlearning_rate=0.05.\nmax_depth=4.")])
s=slide(5,"XGBoost • 2/3","Метрики XGBoost")
scorecards(s,xgb)
s=slide(6,"XGBoost • 3/3","Преимущества и ограничения")
addcards(s,[("Преимущества","Высокие тестовые метрики.\nМеньше разрыв train/test.\nКонтроль сложности."),
            ("Ограничения","Нужна настройка параметров.\nВозможен overfitting.\nНужна валидация и подбор порога.")])
s=slide(7,"Итог","Какую модель использовать?")
panel(s,.85,2.1,11.62,2.6)
label(s,"Модель                 F1 test               ROC-AUC test",1.25,2.42,10.5,.5,21,GRAY,True)
label(s,f"Random Forest         {rf['f1_test']:.3f}                    {rf['roc_auc_test']:.3f}",1.25,3.18,10.5,.4,22,WHITE)
label(s,f"XGBoost                   {xgb['f1_test']:.3f}                    {xgb['roc_auc_test']:.3f}",1.25,3.84,10.5,.4,22,GREEN,True)
winner=max([rf,xgb],key=lambda row:(row["f1_test"],row["roc_auc_test"]))["model"]
label(s,f"Рекомендация по F1 на тесте: {winner}",.85,5.08,11.65,.6,27,BLUE,True)
label(s,"Дальше: кросс-валидация, подбор порога, цена ошибок.",.85,5.95,11.6,.55,20,WHITE)
dest.parent.mkdir(exist_ok=True)
prs.save(dest)
print(f"Saved {dest}, slides: {len(prs.slides)}")
