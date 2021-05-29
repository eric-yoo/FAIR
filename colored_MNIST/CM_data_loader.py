import numpy as np
import matplotlib.pyplot as plt
import wget

class CM_data_loader():
    
    def __init__(self):

        url = "https://drive.google.com/file/d/11K-GmFD5cg3_KTtyBRkj9VBEnHl-hx_Q/view?usp=sharing"
        wget.download(url)

        dat = np.load("colored_mnist.npy", allow_pickle=True, encoding="latin1").item()

        self.train_image = dat['train_image']
        self.train_label = dat['train_label']
        self.test_image  = dat['test_image']
        self.test_label  = dat['test_label']
        
    def visualize(self):
        
        for i in range(3):
            
            plt.imshow(self.train_image[i])
            plt.show()

if __name__=="__main__":
    

    CM = CM_data_loader()
    CM.visualize()


