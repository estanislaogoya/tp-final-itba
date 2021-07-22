import src.features.featEng as fe

def splitDataFramePerStock(df):
    dfs = []
    for i in df.asset.unique():
        df_tmp = df[df['asset']==i]
        dfs.append(fe.pricePredictionFeatureEng40(df_tmp))
    return dfs
