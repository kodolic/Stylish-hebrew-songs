import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from pca import pca
import plotly.graph_objs as go
from matplotlib.lines import Line2D


class utils:
    colorList = ['blue', 'green', 'red', 'purple', 'yellow', 'cyan', 'orange', 'pink']

    @staticmethod
    def saveDict(dictToSave, filename):
        with open(f'{filename}.json', 'w') as file:
            json.dump(dictToSave, file, indent=4)

    @staticmethod
    def readDict(filename):
        with open(filename, 'r') as file:
            loadedDict = json.load(file)
        return loadedDict

    @staticmethod
    def plot3DGraphForAvgClassVector(x, y, z, labels, axises, title=""):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        plt.ion()

        # Create a scatter plot
        scatter = ax.scatter(x, y, z, c=np.arange(len(x)), cmap='tab20', s=75)

        # Create a legend using scatter.legend_elements
        legend_elements, _ = scatter.legend_elements()
        ax.legend(legend_elements, labels, title="Data Items", bbox_to_anchor=(1.05, 1), loc='upper left',
                  fontsize='small')

        ax.set_xlabel(axises[0])
        ax.set_ylabel(axises[1])
        ax.set_zlabel(axises[2])
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot3DGraphByClasses(X, Y, Z, labels, axises, styleMeans, title=""):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(X)):
            ax.scatter(X[i], Y[i], Z[i], c=utils.colorList[i], label=labels[i])

        for i in range(len(styleMeans)):
            ax.scatter(styleMeans[i][0], styleMeans[i][1], styleMeans[i][2], c=utils.colorList[i], s=165)

        ax.set_title(title)
        ax.set_xlabel(axises[0])
        ax.set_ylabel(axises[1])
        ax.set_zlabel(axises[2])

        ax.legend()
        plt.show()

    @staticmethod
    def plot2DGraphByClasses(X, Y, labels, axises, styleMeans, title=""):
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot each class
        for i in range(len(X)):
            ax.scatter(X[i], Y[i], c=utils.colorList[i], label=labels[i])

        for i in range(len(styleMeans)):
            ax.scatter(styleMeans[i][0], styleMeans[i][1], c=utils.colorList[i], s=165)

        # Labels and title
        ax.set_title(title)
        ax.set_xlabel(axises[0])
        ax.set_ylabel(axises[1])

        # Legend
        ax.legend()

        # Show plot
        plt.show()

    @staticmethod
    def plotly3D(X, Y, Z, labels, axises, title=""):
        traces = []
        for i in range(len(X)):
            trace = go.Scatter3d(
                x=X[i],
                y=Y[i],
                z=Z[i],
                mode='markers',
                marker=dict(
                    size=5,
                    color=utils.colorList[i % len(utils.colorList)],
                ),
                name=labels[i]
            )
            traces.append(trace)

        layout = go.Layout(
            title=title,
            scene=dict(
                xaxis_title=axises[0],
                yaxis_title=axises[1],
                zaxis_title=axises[2]
            ),
            legend=dict(
                x=0,
                y=1
            )
        )

        # Create figure and show
        fig = go.Figure(data=traces, layout=layout)
        fig.show()


n2s = utils.readDict('LabelToStyle.json')
s2n = utils.readDict('StyleToLabel.json')


class NormalizationMethods:
    @staticmethod
    def min_max_scaling(series):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return pd.DataFrame(scaler.fit_transform(series), columns=series.columns)

    @staticmethod
    def z_score_normalization(series):
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(series), columns=series.columns)

    @staticmethod
    def robust_scaling(series):
        scaler = RobustScaler()
        return pd.DataFrame(scaler.fit_transform(series), columns=series.columns)

    @staticmethod
    def log_transformation(series):
        return pd.DataFrame(np.log1p(series), columns=series.columns)

    @staticmethod
    def maxabs_scaling(series):
        scaler = MaxAbsScaler()
        return pd.DataFrame(scaler.fit_transform(series), columns=series.columns)

    @staticmethod
    def decimal_scaling(series):
        max_abs = abs(series).max()
        scaling_factor = 10 ** len(str(int(max_abs)))
        return series / scaling_factor

    @staticmethod
    def unit_vector_transformation(series):
        norm = np.linalg.norm(series)
        return series / norm if norm != 0 else series


class DataVisualizer:

    @staticmethod
    def revealPcaFeatures(data, dims=3):
        df = pd.concat(data, axis=0)
        df_labels = df.iloc[:, -1]  # This selects the last column
        unlabeledDf = df.iloc[:, :-1]  # This selects all but the last column

        n_components = min(unlabeledDf.shape[0], dims)
        myPCA = pca(n_components=n_components)

        data_pca = myPCA.fit_transform(unlabeledDf)
        print(data_pca['topfeat'])

        # results = myPCA.results
        # for r in results:
        #     print(results[r])

    @staticmethod
    def activatePCA(data, labels, dims=3):
        df = pd.concat(data, axis=0)
        print(f'number of samples: {df.shape[0]}, number of features: {df.shape[1]}')
        df_labels = df.iloc[:, -1]  # This selects the last column
        unlabeledDf = df.iloc[:, :-1]  # This selects all but the last column

        # print(df_labels)
        n_components = min(unlabeledDf.shape[0], dims)
        regPca = PCA(n_components=n_components)
        data_pca = regPca.fit_transform(unlabeledDf)

        DataVisualizer.plotGraph(data_pca, df_labels, labels, "PCA Projection", dims)

    @staticmethod
    def activateTSNE(data, labels, dims=2):
        # Combine data
        df = pd.concat(data, axis=0)
        print(f'number of samples: {df.shape[0]}, number of features: {df.shape[1]}')

        # Extract labels and features
        df_labels = df.iloc[:, -1]  # This selects the last column
        unlabeledDf = df.iloc[:, :-1]  # This selects all but the last column

        # Initialize t-SNE
        tsne = TSNE(n_components=dims, random_state=0)
        data_tsne = tsne.fit_transform(unlabeledDf)

        DataVisualizer.plotGraph(data_tsne, df_labels, labels, "T-SNE Projection", dims)

    @staticmethod
    def activateLDA(data, labels, dims=2):
        df = pd.concat(data, axis=0)
        print(f'number of samples: {df.shape[0]}, number of features: {df.shape[1]}')

        df_labels = df.iloc[:, -1]
        unlabeledDf = df.iloc[:, :-1]

        n_components = min(unlabeledDf.shape[1], len(labels))
        lda = LDA(n_components=n_components)
        data_lda = lda.fit_transform(unlabeledDf, df_labels)
        DataVisualizer.plotGraph(data_lda, df_labels, labels, "2D Isomap Projection", dims)

    @staticmethod
    def activateIsomap(data, labels, dims=2):
        df = pd.concat(data, axis=0)
        print(f'number of samples: {df.shape[0]}, number of features: {df.shape[1]}')

        df_labels = df.iloc[:, -1]
        unlabeledDf = df.iloc[:, :-1]

        isomap = Isomap(n_components=dims)
        data_isomap = isomap.fit_transform(unlabeledDf)

        DataVisualizer.plotGraph(data_isomap, df_labels, labels, "Isomap Projection", dims)

    @staticmethod
    def plotGraph(data_pca, sampleLabels, labels, title="", dim=2):
        assert dim in [2, 3]
        cList = ['blue', 'green', 'red', 'purple', 'yellow', 'cyan', 'orange', 'pink']
        color_map = {int(s2n[label]): cList[i] for i, label in enumerate(labels)}
        colors = sampleLabels.map(color_map)

        # Create handles for the legend
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[int(s2n[label])], markersize=5) for label
            in labels]
        legend_labels = labels

        fig = plt.figure(figsize=(10, 7))
        ax = None
        if dim == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=colors, s=75)
        else:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=colors, s=75)

        ax.legend(legend_handles, legend_labels, title="Styles", bbox_to_anchor=(1.05, 1), loc='best',
                  fontsize='small')
        ax.set_title(title)
        ax.set_xlabel('d 1')
        ax.set_ylabel('d 2')

        if dim == 3:
            ax.set_zlabel('d 3')

        plt.show()


class FeatureAnalyser:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.path = path

    def editColumn(self, column_name, reverse=True, Percentage=False):
        assert reverse is True and Percentage is False or reverse is False and Percentage is True
        column = self.data[column_name]
        if reverse:
            column = 1 - column
        if Percentage:
            column = 100 - column

        self.data[column_name] = column

    def applyNormalizations(self, dirc='normalizedDataSets'):
        # methods = [NormalizationMethods.min_max_scaling, NormalizationMethods.z_score_normalization,
        #            NormalizationMethods.robust_scaling,
        #            NormalizationMethods.log_transformation, NormalizationMethods.maxabs_scaling,
        #            NormalizationMethods.decimal_scaling,
        #            NormalizationMethods.unit_vector_transformation]
        methods = [NormalizationMethods.min_max_scaling, NormalizationMethods.z_score_normalization,
                   NormalizationMethods.robust_scaling]

        for method in methods:
            print(method.__name__)
            normDf = self.normalizeData(method)
            self.saveToCsv(normDf, f'{dirc}/{method.__name__}.csv')

    def applyLabeling(self):
        mapping = {}
        newColumn = []
        for i, value in enumerate(self.data['Music Style']):
            if type(value) is not float:
                value = value.strip()
                if value.endswith('”},\r\n\u200e'):
                    value = value.removesuffix('”},\r\n\u200e')

                value = FeatureAnalyser.normalizeString(value)

                if value not in mapping.keys() and value.strip() != "" and value is not None:
                    mapping[value] = value
                # print(i, value, type(value))
            else:
                value = "Other"
                mapping[value] = value

            newColumn.append(value)

        self.data['Music Style'] = newColumn

        # print(self.data['Music Style'])

        labels = 0
        for key in mapping.keys():
            mapping[key] = labels
            labels += 1

        inverted_dict = {value: key for key, value in mapping.items()}

        utils.saveDict(inverted_dict, 'LabelToStyle')
        utils.saveDict(mapping, 'StyleToLabel')

        self.data['MappedStyle'] = self.data['Music Style'].map(mapping)

        # self.saveToCsv(self.data, "datasetMappedStyles.csv")

    def normalizeData(self, func):
        columns = self.data.columns[8:]
        # ignoreColumns = ['releaseYear', 'Music Style', 'percentageOfTotalWordsToUnique', 'MappedStyle', 'Birth Year']
        ignoreColumns = ['releaseYear',
                         'Music Style',
                         'percentageOfTotalWordsToUnique',
                         'MappedStyle',
                         'Birth Year',
                         'sentimentScore',
                         'wordCount',
                         'uniqueWords',
                         'DiffPOS',
                         'numberOfBiGrams',
                         'numberOfTriGrams',
                         'averageSetWordLength',
                         'readabilityMeasure', 'positiveWords', 'negativeWords', 'avgSimilarityMeasure',
                         'semantic_similarity', 'heBERT_sentiment']

        dataStyles = self.data['MappedStyle']
        features = []
        for column in columns:
            if column not in ignoreColumns:
                features.append(column)

        featuresDf = self.data.copy()[features]

        if func is not None:
            featuresDf = func(featuresDf)

        featuresDf['MappedStyle'] = dataStyles

        return featuresDf

    def groupingByMusicStyle(self, groupList, func=None, avg=False, batchSize=32):
        df = self.normalizeData(func=func)
        sumGroups = []
        groupNames = []
        grouped = df.groupby('MappedStyle')

        for label, group in grouped:
            currentStyle = n2s[str(label)]
            if currentStyle in groupList:
                print(f"Group {label}, {currentStyle}: {len(group)}")
                if avg:
                    # Calculate the average vector for the group
                    sumVector = group.mean(numeric_only=True).to_frame().T
                    sumVector['MappedStyle'] = label
                    sumGroups.append(sumVector)
                else:
                    # Create batches and calculate average for each batch
                    num_batches = int(np.ceil(len(group) / batchSize))
                    batches = np.array_split(group, num_batches)
                    batch_list = []

                    for batch in batches:
                        batchSum = batch.mean(numeric_only=True).to_frame().T
                        batchSum['MappedStyle'] = label
                        batch_list.append(batchSum)

                    startDF = pd.concat(batch_list, ignore_index=True)
                    sumGroups.append(startDF)

                groupNames.append(currentStyle)

        return sumGroups, groupNames

    # def groupingByMusicStyle(self, groupList, func=None, avg=False):
    #     df = self.normalizeData(func=func)
    #     # print(df.columns)
    #     batchSize = 1
    #     count = 0
    #     sumGroups = []
    #     groupNames = []
    #     grouped = df.groupby('MappedStyle')
    #     for label, group in grouped:
    #         currentStyle = n2s[str(label)]
    #         if currentStyle in groupList:
    #             print(f"Group {label}, {currentStyle}: {len(group)}")
    #             if avg:
    #                 sumVector = group.sum()[0:-1] / len(group)
    #                 sumVector = pd.DataFrame([sumVector.values], columns=sumVector.index)
    #                 # print(type(sumVector), len(sumVector), sumVector.shape)
    #                 sumVector['MappedStyle'] = label
    #                 sumGroups.append(sumVector)
    #             else:
    #                 num_batches = int(np.ceil(len(group) / batchSize))
    #                 batches = np.array_split(group, num_batches)
    #                 startDF = pd.DataFrame(columns=df.index)
    #                 for batch in batches:
    #                     batchSum = batch.sum()[0:-1] / len(batch)
    #                     batchSum = pd.DataFrame([batchSum.values], columns=batchSum.index)
    #                     # print(type(sumVector), len(sumVector), sumVector.shape)
    #                     batchSum['MappedStyle'] = label
    #                     startDF = pd.concat([batchSum, startDF], axis=0, ignore_index=True)
    #                 sumGroups.append(startDF)
    #                 # sumGroups.append(group)
    #             groupNames.append(n2s[str(label)])
    #
    #     return sumGroups, groupNames

    # def groupingByMusicStyle(self, groupList, func=None, avg=False):
    #     df = self.normalizeData(func=func)
    #     # print(df.columns)
    #     sumGroups = []
    #     groupNames = []
    #     grouped = df.groupby('MappedStyle')
    #     for label, group in grouped:
    #         currentStyle = n2s[str(label)]
    #         if currentStyle in groupList:
    #             print(f"Group {label}, {currentStyle}: {len(group)}")
    #             if avg:
    #                 sumVector = group.sum()[0:-1] / len(group)
    #                 sumVector = pd.DataFrame([sumVector.values], columns=sumVector.index)
    #                 # print(type(sumVector), len(sumVector), sumVector.shape)
    #                 sumVector['MappedStyle'] = label
    #                 sumGroups.append(sumVector)
    #             else:
    #                 sumGroups.append(group)
    #             groupNames.append(n2s[str(label)])
    #
    #     return sumGroups, groupNames

    def SyntaxComplexity(self):
        X_features = []
        Y_features = []
        for row in self.data.itertuples():
            # Y_features.append(
            #     0.7 * getattr(row, 'readabilityMeasure') + 0.3 * getattr(row, 'averageSetWordLength') + 0.01 * getattr(
            #         row, "bigramsEntropy") + 0.01 * getattr(row, "trigramsEntropy"))

            Y_features.append(
                0.3 * getattr(row, 'theUniquenessLvlOfTheRepeatedSongs') + 0.3 * getattr(row,
                                                                                         'avg_word_similarity_hebrew')
                + 0.3 * getattr(row, "avg_word_similarity_english"))

            # Y_features.append(
            #     0.50 * getattr(row, 'average_word_frequency') + 0.30 * getattr(row, 'avg_word_similarity_hebrew') + 0.10 * getattr(
            #         row, "ratioOfTotalWordsToUnique") + 0.05 * getattr(row, "numberOfRepeatedWords") + 0.05 * getattr(row, "DiffLemmas"))

            X_features.append(getattr(row, 'releaseYear'))

        return X_features, Y_features

    def songLength(self):
        X_features = []
        Y_features = []
        for row in self.data.itertuples():
            Y_features.append(
                0.7 * getattr(row, 'wordCount') + 0.3 * getattr(row, 'averageSetWordLength') + 1 * getattr(row,
                                                                                                           'numberOfRepeatedWords'))
            X_features.append(getattr(row, 'releaseYear'))

        return X_features, Y_features

    def VocabRichness(self):
        X_features = []
        Y_features = []
        for row in self.data.itertuples():
            Y_features.append(
                0.2 * getattr(row, 'avgSimilarityMeasure') + 0.8 * getattr(row, 'RatioOfPOStoWords') + 0.1 * getattr(
                    row, 'numberOfRepeatedWords') + 0.98 * getattr(row, 'averageSetWordLength'))
            X_features.append(getattr(row, 'releaseYear'))

        return X_features, Y_features

    @staticmethod
    def saveToCsv(data, filename):
        data.to_csv(filename, encoding='utf-8-sig')

    @staticmethod
    def normalizeString(s):
        elements = [element.strip() for element in s.split(',')]
        elements.sort()
        return ', '.join(elements)

    def analyseStyle(self, genre, axis=[], func=None, avg=False, batchSize=32, reveal=False):
        groupedData, retLabels = self.groupingByMusicStyle(groupList=genre,
                                                           func=func,
                                                           avg=avg,
                                                           batchSize=batchSize)

        groupMeans = []

        if reveal:
            DataVisualizer.revealPcaFeatures(groupedData, dims=3)
            return

        # DataVisualizer.activatePCA(groupedData, retLabels, dims=3)
        # DataVisualizer.activatePCA(groupedData, retLabels, dims=2)
        # DataVisualizer.activateTSNE(groupedData, retLabels, dims=2)
        # DataVisualizer.activateTSNE(groupedData, retLabels, dims=3)
        # DataVisualizer.activateIsomap(groupedData, retLabels, dims=2)

        X, Y, Z = [], [], []

        for g in groupedData:
            X.append([*g[axis[0]].values])
            Y.append([*g[axis[1]].values])
            Z.append([*g[axis[2]].values])

        for x, y, z in zip(X, Y, Z):
            groupMeans.append([np.mean(x), np.mean(y), np.mean(z)])

        utils.plot3DGraphByClasses(X, Y, Z,
                                   labels=retLabels,
                                   axises=axis,
                                   styleMeans=groupMeans)

        utils.plot2DGraphByClasses(X, Y,
                                   labels=retLabels,
                                   axises=axis[0:-1],
                                   styleMeans=groupMeans)

        # Remove comment for 3D interactive graph below
        # utils.plotly3D(X, Y, Z, labels=styles, axises=axis)


def plot_data(X, Y, X_title, Y_title, plot_title="Data Plot"):
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, marker='o', linestyle='-', color='blue')
    plt.xlabel(X_title)
    plt.ylabel(Y_title)
    plt.title(plot_title)
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate X-axis labels if needed
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()


if __name__ == "__main__":
    fa = FeatureAnalyser("FixedLatestDataset.csv")
    # pcaModule = PcaOperator()

    # fa.applyNormalizations()
    # fa.applyLabeling()

    # styles = ["Pop", "Mizrahi", "Hip-Hop", "Rock", "Classical, Progressive Rock"]
    # styles = ["Classical, Progressive Rock", "Hip-Hop"]
    # axes = ['WordsRhymes', 'wordCount', 'AvgUniqueness']
    # axes = ['numberOfBiGrams', 'sentimentScore', 'ratioOfTotalWordsToUnique']
    # axes = ['bigramsEntropy', 'sentimentScore', 'ratioOfTotalWordsToUnique']

    # axes = ['trigramsEntropy', 'AvgUniqueness', 'theUniquenessLvlOfTheRepeatedSongs']
    # axes = ['bigramsEntropy', 'sentimentScore', 'average_word_frequency']
    # axes = ["DiffLemmas", "avg_word_similarity_english", "avg_word_similarity_hebrew"]

    axes = ["bigramsEntropy", "ratioOfTotalWordsToUnique", "average_word_frequency"]
    styles = ["Pop", "Rock"]
    fa.analyseStyle(genre=styles,
                    axis=axes,
                    func=NormalizationMethods.z_score_normalization,
                    avg=False,
                    batchSize=64,
                    reveal=True)

    # -------------------------------------------------------------------------------- #

    axes = ["numberOfRepeatedWords", "percentageOfRepeatedWords", "average_word_frequency"]
    styles = ["Mizrahi, Pop", "Folk, Pop", "Pop"]
    fa.analyseStyle(genre=styles,
                    axis=axes,
                    func=NormalizationMethods.min_max_scaling,
                    avg=False,
                    batchSize=128,
                    reveal=True)

    # -------------------------------------------------------------------------------- #

    axes = ["numberOfRepeatedWords", "avg_word_similarity_hebrew", "average_word_frequency"]
    styles = ["Metal, Rock", "Pop, R&B"]
    fa.analyseStyle(genre=styles,
                    axis=axes,
                    func=NormalizationMethods.min_max_scaling,
                    avg=False,
                    batchSize=4,
                    reveal=True)

    # -------------------------------------------------------------------------------- #
    from scipy.ndimage import gaussian_filter1d

    X, Y = fa.SyntaxComplexity()

    df = pd.DataFrame({'Year': X, 'Value': Y})
    df['Year'] = df['Year'] - (df['Year'] % 5)
    df = df.sort_values('Year')
    # df_avg = df.groupby('Year', as_index=False)['Value'].mean()
    groupedDF = pd.DataFrame(df.groupby('Year', as_index=False)['Value'].median())
    newValues = gaussian_filter1d(groupedDF.values, 10)
    newValues = (newValues - newValues.mean()) / newValues.std()
    newValues = newValues + abs(newValues.min())
    groupedDF['Value'] = newValues
    plot_data(groupedDF['Year'], groupedDF['Value'], "Years", "Syntax Complexity Level")

    # X, Y = fa.songLength()
    #
    # df2 = pd.DataFrame({'Year': X, 'Value': Y})
    # df2['Year'] = df2['Year'] - (df2['Year'] % 5)
    # df_avg2 = df2.groupby('Year', as_index=False)['Value'].mean()
    # print(df_avg2)
    # plot_data(df_avg2['Year'], df_avg2['Value'], "Years", "Length of the songs and vocab richness over the years")
    #
    # X, Y = fa.VocabRichness()
    #
    # df2 = pd.DataFrame({'Year': X, 'Value': Y})
    # df2['Year'] = df2['Year'] - (df2['Year'] % 5)
    # df_avg2 = df2.groupby('Year', as_index=False)['Value'].median()
    # print(df_avg2)
    # plot_data(df_avg2['Year'], df_avg2['Value'], "Years", "Vocab richness over the years")
