# This function expects data to be a list/tuple of length 2 with samples and labels: (X, y)
# Both the samples and labels are split into k chunks
#
# The returned partition is a tuple of length k. Each tuple entry is a list (X, y)
# of samples and labels.
#
# TODO: extend to support non-IID
def split_data(data, k):
    partition = []
    data = (data[0][:5000], data[1][:5000])             #    !!!!!!!!!!!!! Chopped the data set size during development
    n = len(data[0])
    splits = [(n // k) * i for i in range(k)]

    for i in range(len(splits)):
        if i == len(splits) - 1:
            partition.append((data[0][splits[i]:, :],
                              data[1][splits[i]:]))
        else:
            partition.append((data[0][splits[i]:splits[i + 1], :],
                              data[1][splits[i]:splits[i + 1]]))
    return partition