
def normalize_x_sf(df, scale=1e4):
    sf = df.sum(axis=0) / scale
    return sf.values
