from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioAnalysis
dir_list = ["./data/Away", "./data/Close", "./data/no_gesture"]

aT.featureAndTrain(dir_list, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knn3", True)

print(aT.fileClassification("./test/Awaytest001.wav", "knn3", "knn"))

# aT.featureAndTrain(dir_list, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnClasses")

# audioAnalysis.fileSpectrogramWrapper("./test/Close001.wav")

# import matplotlib.pyplot as plt
# from pyAudioAnalysis import audioBasicIO
# from pyAudioAnalysis import audioFeatureExtraction
#
# [Fs, x] = audioBasicIO.readAudioFile("./data/away/away009.wav");
# F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
# plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]);
# plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()
