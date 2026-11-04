"""Reproducible four-model churn comparison, using one split and single-thread benchmarks."""
import json, pickle, platform, statistics
from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
OUT.mkdir(parents=True,exist_ok=True)
MODELS=("Random Forest","XGBoost","LightGBM","CatBoost")
CATCOLS=["region","tariff"]
REPEATS=5

def make_prep():
    return ColumnTransformer(
        [("cat",OneHotEncoder(handle_unknown="ignore"),CATCOLS)],
        remainder="passthrough")

def make_model(name):
    if name=="Random Forest":
        return make_pipeline(make_prep(),RandomForestClassifier(
            n_estimators=100,random_state=42,n_jobs=1))
    if name=="XGBoost":
        return make_pipeline(make_prep(),XGBClassifier(
            n_estimators=300,learning_rate=.05,max_depth=4,
            random_state=42,n_jobs=1,eval_metric="logloss"))
    if name=="LightGBM":
        return make_pipeline(make_prep(),LGBMClassifier(
            n_estimators=300,learning_rate=.05,max_depth=4,num_leaves=15,
            random_state=42,n_jobs=1,verbosity=-1))
    if name=="CatBoost":
        return CatBoostClassifier(iterations=300,learning_rate=.05,depth=4,
            random_seed=42,cat_features=CATCOLS,thread_count=1,
            verbose=False,allow_writing_files=False,loss_function="Logloss")
    raise ValueError(name)

def load_data():
    df=pd.read_csv(ROOT/"o_churn_synthetic.csv")
    needed={"subscriber_code","churn","region","tariff"}
    if needed-set(df):
        raise ValueError(f"Missing columns: {needed-set(df)}")
    X=df.drop(columns=["churn","subscriber_code"])
    y=df["churn"].astype(int)
    return (df,*train_test_split(X,y,test_size=.2,stratify=y,random_state=42))

def benchmark(name,X_train,X_test,y_train,y_test):
    times=[]
    for i in range(REPEATS+1):
        model=make_model(name)
        t0=perf_counter()
        model.fit(X_train,y_train)
        times.append(perf_counter()-t0)
    # Discard first training run; median of 5 freshly fitted models.
    fit_time=statistics.median(times[1:])
    model.predict_proba(X_test) # warmup
    itimes=[]
    for i in range(REPEATS):
        t0=perf_counter()
        probability=model.predict_proba(X_test)[:,1]
        itimes.append(perf_counter()-t0)
    train_pred=model.predict(X_train)
    test_pred=model.predict(X_test)
    row={
        "model":name,
        "f1_train":float(f1_score(y_train,train_pred,zero_division=0)),
        "f1_test":float(f1_score(y_test,test_pred,zero_division=0)),
        "roc_auc_test":float(roc_auc_score(y_test,probability)),
        "train_seconds_median":float(fit_time),
        "inference_seconds_median":float(statistics.median(itimes)),
        "model_size_kb":float(len(pickle.dumps(model))/1024),
    }
    row["f1_gap"]=row["f1_train"]-row["f1_test"]
    print(name, {k:round(v,4) if isinstance(v,float) else v for k,v in row.items()})
    safe=name.lower().replace(" ","_")
    chart=ConfusionMatrixDisplay.from_predictions(y_test,test_pred,cmap="Blues")
    chart.ax_.set_title(name)
    chart.figure_.savefig(OUT/(safe+"_confusion.png"),dpi=140)
    plt.close(chart.figure_)
    return row

def write_outputs(df,X_train,X_test,results):
    table=pd.DataFrame(results)
    table.to_csv(OUT/"metrics.csv",index=False,float_format="%.8f")
    boosters=[r for r in results if r["model"]!="Random Forest"]
    leader_auc=max(boosters,key=lambda r:r["roc_auc_test"])
    leader_f1=max(boosters,key=lambda r:r["f1_test"])
    payload={
       "dataset":"o_churn_synthetic.csv",
       "n_rows":len(df),"train_rows":len(X_train),"test_rows":len(X_test),
       "random_state":42,"n_features_raw":X_train.shape[1],
       "class_distribution":{str(k):int(v) for k,v in df["churn"].value_counts().items()},
       "benchmark":{"n_jobs":1,"warmup_fits":1,"fit_repeats":5,
                    "inference_repeats":5,"inference_rows":len(X_test),
                    "size_definition":"pickle full fitted estimator or pipeline"},
       "versions":{p:version(p) for p in ("scikit-learn","xgboost","lightgbm","catboost")},
       "metrics":results,
       "best_boost_auc":leader_auc["model"],
       "best_boost_f1":leader_f1["model"]}
    (OUT/"metrics.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    ax=table.set_index("model")[["f1_train","f1_test","roc_auc_test"]].plot.bar(
       figsize=(10,5),rot=0,ylim=(0,1.1),title="ML-P-8 churn models")
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(OUT/"comparison.png",dpi=150)
    plt.close()
    lines=["| Модель | F1 train | F1 test | ROC-AUC test | Обучение, с | Инференс, с | Размер, КБ | Δ F1 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['f1_train']:.3f} | {r['f1_test']:.3f} | "
         f"{r['roc_auc_test']:.3f} | {r['train_seconds_median']:.3f} | "
         f"{r['inference_seconds_median']:.5f} | {r['model_size_kb']:.1f} | {r['f1_gap']:.3f} |")
    verdict=(f"**Лучший бустинг по ROC-AUC test:** {leader_auc['model']} "
       f"({leader_auc['roc_auc_test']:.3f}). **Лучший по F1 test:** "
       f"{leader_f1['model']} ({leader_f1['f1_test']:.3f}).")
    report=("\n".join(["# Фактические результаты по CSV","","3000 абонентов, train 2400, test 600,",
                       "единый random_state=42 и stratify=y.",""]+lines)+
        "\n\n"+verdict+"\n\nОдно ядро; 5 замеров после исключения первого,"
        " медиана обучения. Инференс — 5 повторов для 600 строк."
        " Размер модели: pickle обученной модели/Pipeline."
        " CatBoost — native region/tariff; остальные — one-hot.\n")
    (OUT/"report.md").write_text(report,encoding="utf-8")
    readme=ROOT/"README.md"
    if readme.exists():
        s=readme.read_text(encoding="utf-8")
        a,b="<!-- METRICS_START -->","<!-- METRICS_END -->"
        if a in s and b in s:
            generated=("\n### Результаты по исходному CSV\n\n"+
                "\n".join(lines)+"\n\n"+verdict+
                "\n\n**Время:** 1 поток, первый запуск отброшен, медиана следующих 5. "+
                "**Инференс:** 600 строк, 5 повторов. **Размер:** pickle full model/Pipeline. "+
                "**Категории:** native в CatBoost, one-hot в остальных моделях.\n")
            left=s.index(a)+len(a)
            right=s.index(b,left)
            readme.write_text(s[:left]+generated+"\n"+s[right:],encoding="utf-8")
    return payload

def main():
    df,X_train,X_test,y_train,y_test=load_data()
    print("train:",X_train.shape,"test:",X_test.shape)
    results=[benchmark(name,X_train,X_test,y_train,y_test) for name in MODELS]
    write_outputs(df,X_train,X_test,results)
    return pd.DataFrame(results).set_index("model")

if __name__=="__main__":
    main()
