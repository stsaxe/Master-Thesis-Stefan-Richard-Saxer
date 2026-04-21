import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

plt.style.use('default')

FONTSIZE_LABEL = 15
FONTSIZE_TICKS = 15
FONTSIZE_TITLE = 20


def plotReport(y_test, y_test_pred, labels, show: bool = True, name: str = None, savePath: str = None, digits: int = 4):
    plt.style.use('default')
    report = classification_report(y_test, y_test_pred, labels=labels, digits=digits, zero_division=0.0)

    prefix = "Classification Report"

    if show:
        if name is None:
            print()
        else:
            print(prefix + " - " + name)

        print(report)
        print("\n")

    if savePath is not None and name is not None:
        report = classification_report(y_test, y_test_pred, labels=labels, digits=digits, zero_division=0.0,
                                       output_dict=True)

        report = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Label'}).round(3)

        report_lower = report.iloc[:-3, :]
        report_empty = pd.DataFrame([['' for i in report.columns]], columns=report.columns)
        report_upper = report.iloc[-3:,]

        report = pd.concat([report_lower, report_empty, report_upper])

        report.to_csv(savePath + prefix + " - " + name + ".csv", encoding='utf-8', index=False)


def plotTree(model, name: str, features_names: list[str], savePath: str = None, sizeFactor=1, dpi: int = 100,
             show=True):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16 * sizeFactor, 9 * sizeFactor), dpi=dpi)
    plt.suptitle(name, fontsize=25 * sizeFactor)
    fig.tight_layout()

    tree.plot_tree(model, feature_names=features_names, class_names=model.classes_, ax=ax, filled=True)

    if savePath is not None:
        fig.savefig(savePath + name, dpi=500)

    if show:
        plt.show()

    plt.close()


def plotMatrix(y_test, y_test_pred, labels, dpi: int = 300, normalize: str = 'true', savePath: str = None,
               name: str = None, show: bool = True, digits: int = 2):

    cm = confusion_matrix(y_test, y_test_pred, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(11, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=True,
        values_format=".2f",
        include_values=True,
        text_kw={"fontsize": 6}
    )

    plt.style.use('default')


    prefix = "Confusion Matrix"

    if name is None:
        ax.set_title(prefix, fontsize=FONTSIZE_TITLE)
    else:
        ax.set_title(prefix + " - " + name, fontsize=FONTSIZE_TITLE)

    plt.xticks(fontsize=FONTSIZE_TICKS)
    plt.yticks(fontsize=FONTSIZE_TICKS)

    plt.xlabel('Predicted Label', fontsize=FONTSIZE_LABEL)
    plt.ylabel('True Label', fontsize=FONTSIZE_LABEL)

    fig.tight_layout()

    if name is not None and savePath is not None:
        fig.savefig(savePath + prefix + " - " + name, dpi=300)

    if show:
        plt.show()

    plt.close()


def featureImportanceDataFrame(features: np.array, importance: np.array, other: bool = False, rel_limit: float = 1,
                               abs_limit: int = 10):
    plt.style.use('default')
    assert len(features) == len(importance)
    assert len(features) > 0

    frequency = pd.DataFrame(features, columns=['Feature'])
    frequency['Importance'] = importance
    frequency['Importance'] *= 100

    frequency.sort_values(by='Importance', ascending=False, inplace=True)
    frequency.reset_index(inplace=True, drop=True)

    if other:
        otherName = 'Other Features'
        minFeatureImportance = frequency['Importance'].min()
        cutOffIndex = 0

        if len(importance) <= abs_limit and minFeatureImportance > rel_limit:
            return frequency

        elif len(importance) > abs_limit and minFeatureImportance > rel_limit:
            cutOffIndex = abs_limit - 1

        elif len(importance) <= abs_limit and minFeatureImportance <= rel_limit:
            cutOffIndex = frequency.index[frequency['Importance'] <= rel_limit][0]

        elif len(importance) > abs_limit and minFeatureImportance <= rel_limit:
            cutOffIndex = min(abs_limit - 1, frequency.index[frequency['Importance'] <= rel_limit][0])

        frequency.loc[cutOffIndex:, "Feature"] = otherName
        frequency = frequency.groupby("Feature").sum().sort_values(by='Importance', ascending=False).reset_index()

    return frequency


def plotFeatureImportance(features, importance, dpi: int = 300, rel_limit: float = 1, abs_limit: int = 10,
                          savePath: str = None, name: str = None, show: bool = True):
    plt.style.use('default')
    frequency = featureImportanceDataFrame(np.array(features), np.array(importance), rel_limit=rel_limit, other=True,
                                           abs_limit=abs_limit)

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    bars = plt.barh(frequency['Feature'], frequency['Importance'])

    if 'Other Features' in list(frequency['Feature'].unique()):
        indexOther = frequency.index[frequency['Feature'] == 'Other Features'][0]
        bars[indexOther].set_color('m')

    plt.gca().invert_yaxis()

    plt.xlabel('relative Feature Importance', fontsize=FONTSIZE_LABEL)

    prefix = "Feature Importance"

    if name is None:
        plt.title(prefix, fontsize=FONTSIZE_TITLE)
    else:
        plt.title(prefix + " - " + name, fontsize=FONTSIZE_TITLE)

    plt.gca().xaxis.set_major_formatter(PercentFormatter())

    plt.xticks(fontsize=FONTSIZE_TICKS)
    plt.yticks(fontsize=FONTSIZE_TICKS)

    fig.tight_layout()

    if savePath is not None and name is not None:
        fig.savefig(savePath + prefix + " - " + name, dpi=300)

    if show:
        plt.show()

    plt.close()


def plot_adv_type_distribution(dataset: pd.DataFrame, adv_column: str, name: str, show: bool = True,
                               savePath: str = None):
    plt.style.use('default')
    temp = dataset.copy(deep=True)

    temp[adv_column] = temp[adv_column].replace({"": "None"})
    temp[adv_column] = temp[adv_column].str.split(',', expand=False)
    temp = temp.explode(adv_column)

    temp = temp.groupby(adv_column).size().reset_index().rename(columns={0: 'Frequency'})
    temp['Frequency'] = temp['Frequency'] / temp['Frequency'].sum() * 100

    df = temp[[adv_column, 'Frequency']]
    df.sort_values(by="Frequency", ascending=True, inplace=True)

    plt.figure(figsize=(16, 9), dpi=300)
    plt.gca().xaxis.set_major_formatter(PercentFormatter())

    plt.barh(df[adv_column], df['Frequency'])

    plt.xticks(fontsize=FONTSIZE_TICKS)
    plt.yticks(fontsize=FONTSIZE_TICKS)

    plt.xlabel("Relative Occurrences", fontsize=FONTSIZE_LABEL)

    plt.xlim(left=0)

    plt.title(name, fontsize=FONTSIZE_TITLE)

    plt.tight_layout()

    if not savePath is None:
        plt.savefig(savePath + name, dpi=300)

    if show:
        plt.show()

    plt.close()


def plot_adv_count_distribution(dataset: pd.DataFrame, adv_column: str, name: str, show: bool = True,
                                savePath: str = None):
    plt.style.use('default')
    temp = dataset.copy(deep=True)

    temp[adv_column] = temp[adv_column].replace({"": "None"})
    temp[adv_column] = temp[adv_column].str.split(',', expand=False)

    temp = temp[temp[adv_column].isna() == False]

    temp['Length'] = temp[adv_column].apply(lambda x: len(x))
    distribution = (temp.groupby("Length").size() / len(temp) * 100).reset_index().rename(columns={0: 'Frequency'})

    plt.figure(figsize=(16, 9), dpi=300)
    plt.bar(distribution["Length"], distribution['Frequency'])

    plt.xticks(distribution["Length"], fontsize=FONTSIZE_TICKS)
    plt.yticks(fontsize=FONTSIZE_TICKS)

    plt.ylim(bottom=0)
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.ylabel("Relative Frequency", fontsize=FONTSIZE_LABEL)
    plt.xlabel("Number of Advertising Data Structures", fontsize=FONTSIZE_LABEL)

    plt.title(name, fontsize=FONTSIZE_TITLE)

    plt.tight_layout()

    if not savePath is None:
        plt.savefig(savePath + name, dpi=300)

    if show:
        plt.show()

    plt.close()


def plot_hist_distribution(values, x_label: str, name: str, show: bool = True, bins: int = 15, savePath: str = None):
    plt.style.use('default')
    plt.figure(figsize=(16, 9), dpi=300)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.hist(values, bins=bins, density=True, histtype='step')

    plt.ylabel("Probability Density", fontsize=FONTSIZE_LABEL)
    plt.xlabel(x_label, fontsize=FONTSIZE_LABEL)

    plt.xticks(fontsize=FONTSIZE_TICKS)
    plt.yticks(fontsize=FONTSIZE_TICKS)

    plt.ylim(bottom = 0)
    plt.xlim(left=0)

    plt.title(name, fontsize=FONTSIZE_TITLE)
    plt.tight_layout()

    if not savePath is None:
        plt.savefig(savePath + name, dpi=300)
    if show:
        plt.show()

    plt.close()


def plot_pdu_distribution(dataset: pd.DataFrame, pdu_column: str, name: str, show: bool = True, savePath: str = None):
    plt.style.use('default')
    temp = dataset.groupby(pdu_column).size().reset_index().rename(columns={0: 'Frequency'})
    temp = temp[~temp[pdu_column].str.contains("Malformed")]

    temp.sort_values(by="Frequency", ascending=True, inplace=True)

    temp['Frequency'] /= temp['Frequency'].sum()

    plt.figure(figsize=(16, 9), dpi=300)
    plt.gca().xaxis.set_major_formatter(PercentFormatter())

    plt.barh(temp[pdu_column], temp['Frequency'] * 100)

    plt.xticks(fontsize=FONTSIZE_TICKS)
    plt.yticks(fontsize=FONTSIZE_TICKS)

    plt.xlim(left=0)
    plt.xlabel("Relative Occurrences", fontsize=FONTSIZE_LABEL)

    plt.title(name, fontsize=FONTSIZE_TITLE)
    plt.tight_layout()

    if not savePath is None:
        plt.savefig(savePath + name, dpi=300)
    if show:
        plt.show()

    plt.close()
