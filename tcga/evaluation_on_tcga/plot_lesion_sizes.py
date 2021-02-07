import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from evaluation_metrics import get_FROC_fn


if __name__ == "__main__":

    tcga_info = pd.read_csv('../filtered_tcga_annotations/tcga_info_fake_recist_0.5_350.csv')
    
    num_sizes = []
    

    filtered_dl_info = tcga_info.loc[(tcga_info['length']) < 10]
    index = filtered_dl_info.index
    num_sizes.append(len(index))
    print(len(index))

    filtered_dl_info = tcga_info.loc[((tcga_info['length'] >= 10) & (tcga_info['length'] <= 30))]
    index = filtered_dl_info.index
    num_sizes.append(len(index))
    print(len(index))

    filtered_dl_info = tcga_info.loc[((tcga_info['length'] > 30) & (tcga_info['length'] <= 50))]
    index = filtered_dl_info.index
    num_sizes.append(len(index))
    print(len(index))

    filtered_dl_info = tcga_info.loc[((tcga_info['length'] > 50) & (tcga_info['length'] <= 70))]
    index = filtered_dl_info.index
    num_sizes.append(len(index))
    print(len(index))

    filtered_dl_info = tcga_info.loc[((tcga_info['length'] > 70) & (tcga_info['length'] <= 100))]
    index = filtered_dl_info.index
    num_sizes.append(len(index))
    print(len(index))

    filtered_dl_info = tcga_info.loc[(tcga_info['length']) > 100]
    index = filtered_dl_info.index
    num_sizes.append(len(index))
    print(len(index))

    
    lesion_sizes = ("< 10", "10 - 30", "30 - 50", "50 - 70", "70 - 100", "> 100")
    y_pos = np.arange(len(lesion_sizes))
 
    plt.bar(y_pos, num_sizes)

    plt.xticks(y_pos, lesion_sizes)
    plt.ylabel('Anzahl Läsionen')
    plt.title('Anzahl Läsionen mit verschiedenen Durchschnittsgrößen')

    plt.show()




