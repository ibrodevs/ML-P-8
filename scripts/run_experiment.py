"""Independent preprocessing and a fair model comparison on original CSV."""
import json
from pathlib import Path
from time import perf_counter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
OUT.mkdir(exist_ok=True)

def make_prep():
    return ColumnTransformer([
        ("cat",OneHotEncoder(handle_unknown="ignore"),["region","tariff"])
    ], remainder="passthrough")

def fit_and_report(name,model,X_train,X_test,y_train,y_test):
    pipe=make_pipeline(make_prep(),model)
    start=perf_counter()
    pipe.fit(X_train,y_train)
    elapsed=perf_counter()-start
    train_preds=pipe.predict(X_train)
    preds=pipe.predict(X_test)
    row={"model":name,
         "f1_train":float(f1_score(y_train,train_preds)),
         "f1_test":float(f1_score(y_test,preds)),
         "roc_auc_test":float(roc_auc_score(y_test,pipe.predict_proba(X_test)[:,1])),
         "fit_seconds":float(elapsed)}
    row["f1_gap"]=row["f1_train"]-row["f1_test"]
    print(f"{name:14s} F1 train={row['f1_train']:.3f} F1 test={row['f1_test']:.3f} "
          f"AUC test={row['roc_auc_test']:.3f} time={elapsed:.2f}s")
    display=ConfusionMatrixDisplay.from_predictions(y_test,preds,cmap="Blues")
    display.ax_.set_title(name)
    display.figure_.savefig(OUT/("rf" if name=="Random Forest" else "xgboost")+"_confusion.png",dpi=150)
    plt.close(display.figure_)
    return row

def main():
    df=pd.read_csv(ROOT/"o_churn_synthetic.csv")
    X=df.drop(columns=["churn","subscriber_code"])
    y=df["churn"].astype(int)
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,stratify=y,random_state=42)
    print("train:",X_train.shape,"test:",X_test.shape)
    results=[
        fit_and_report("Random Forest",RandomForestClassifier(
            n_estimators=100,random_state=42,n_jobs=1),X_train,X_test,y_train,y_test),
        fit_and_report("XGBoost",XGBClassifier(
            n_estimators=300,learning_rate=0.05,max_depth=4,
            random_state=42,n_jobs=1,eval_metric="logloss"),X_train,X_test,y_train,y_test)
    ]
    dfres=pd.DataFrame(results)
    dfres.to_csv(OUT/"metrics.csv",index=False)
    winner=max(results,key=lambda row:(row["f1_test"],row["roc_auc_test"]))["model"]
    payload={"dataset":"o_churn_synthetic.csv","n_rows":len(df),
             "train_rows":len(X_train),"test_rows":len(X_test),
             "n_features_raw":X.shape[1],
             "class_distribution":{str(k):int(v) for k,v in y.value_counts().items()},
             "metrics":results,"recommended_by_f1_test":winner}
    (OUT/"metrics.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    ax=dfres.set_index("model")[["f1_train","f1_test","roc_auc_test"]].plot.bar(figsize=(9,5),rot=0)
    ax.set_ylim(0,1.12)
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(OUT/"comparison.png",dpi=160)
    plt.close()
    lines=["# Результаты ML-P-8","",f"Train: {len(X_train)}; test: {len(X_test)}; stratified split 80/20; seed 42.","",
           "| Модель | F1 train | F1 test | ROC-AUC test | Время, с | Разрыв F1 |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['f1_train']:.3f} | {r['f1_test']:.3f} | {r['roc_auc_test']:.3f} | {r['fit_seconds']:.3f} | {r['f1_gap']:.3f} |")
    lines += ["",f"**Лучший по F1 test: {winner}.**","",
              "Для производства проверьте кросс-валидацию, подбор порога и стоимость ошибок."]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Recommended:",winner)

if __name__=="__main__":
    main()
