import os
import sys
import numpy as np
#import matplotlib.pyplot as plt
import pylab as plt
import matplotlib.patches as mpatches
import scipy.optimize as opt
from sklearn.svm import SVC

import argparse
import csv

import argparse
import csv



def getArguments():
    parser = argparse.ArgumentParser(description='Import data from a csv file with headers on each column, and plot something...')

    # Command line flags
    parser.add_argument('-c', '--data_csv', action='store', dest='dataFile', help='The input csv data file')
    parser.add_argument('-p', '--plot_folder', action='store', dest='plotFolder', help='The output folder for the plots, must exist')
    parser.add_argument('-k', '--key_classifier', action='store', dest='classifier', help='The classifier to plot against for scatter plots')
    parser.add_argument('-n', '--number_train', action='store', dest='numTrain', help='The number of data points to use in training, the rest will be used for cross validation')
    parser.add_argument('-r', '--regularization_value', action='store', dest='reg', help='The value for the regularization parameter')

    parser.add_argument('-d', '--dimensions', action='store', dest='dimension', help='Adding dimensions for fitting')

    parser.add_argument('-l', '--logistic', action='store_true', dest='logistic', help='Set logistic regression, for classification by the key and not fitting')
    parser.add_argument('-t', '--type', action='store', dest='type', help='To be used with mode -l, set type = linear or = rbf')
    parser.add_argument('--classification', action='store_true', dest='classification', help='For now, if there is just a 0,1 classification scheme, produce scatter plots')
    parser.add_argument('--normalize', action='store_true', dest='normalize', help='For now, if there is just a 0,1 classification scheme, produce scatter plots')
    parser.add_argument('--do_pca', action='store_true', dest='doPCA', help='For now, if there is just a 0,1 classification scheme, produce scatter plots')

    args_ = parser.parse_args()
    return args_


def checkFile(file_):
    # Check that CSV file exists
    if not os.path.isfile(file_):
        print "Error: File %s does not exist." % file_
        print "Exiting with status 1."
        sys.exit(1)

def normalizeData(X):
    meanOfData=X.mean(axis=0)
    normalizedData=X-meanOfData
    stdOfData=normalizedData.std(axis=0)
    normalizedData=normalizedData/stdOfData
    return normalizedData, meanOfData, stdOfData


def ScatterWithLinearSVC(X_,class_,C_,TrainedSVC,Labels_):
    ##"Train" the data, aka fit with vectorial distance to a line, since we are using a linear kernel
    ## Here we essentially minimize the total distance that each point is to a decision boundary
    ## Based on which side of the boundary the point lies (if distance is "positive" or "negative"), the decision is made
    #TrainedSVC = SVC(C = C_, kernel = 'linear').fit(X_,class_)
    if np.shape(X_)[1] >2:
        print "X is larger than 2!"
    Xvalues=X_[:,0]
    Yvalues=X_[:,1]
    xmax, ymax, xmin,ymin =Xvalues.max(), Yvalues.max(), Xvalues.min(), Yvalues.min()
    binning=100.
    ### Make a big grid of coordinates, as a matrix of X positions and a matrix of y positions
    X, Y = np.meshgrid(np.arange(xmin,xmax, (xmax-xmin)/binning),
                       np.arange(ymin,ymax, (ymax-ymin)/binning))
    ### Ravel the X and Y matrices up into vectors, put them together, and feed them to the predict function
    GridPredictions=TrainedSVC.predict(np.c_[X.ravel(), Y.ravel()])
    ### Re-form a matrix of Grid predictions
    GridPredictions=np.reshape(GridPredictions,np.shape(X))
    ### Plot a contour, with coloring each area, fading to 1/3 of the color "strength"
    fig=plt.figure()
    plt.contourf(X,Y,GridPredictions,alpha=0.33)
    plt.scatter(Xvalues, Yvalues,c=class_)   
    plt.scatter(Xvalues, Yvalues,c=class_)   
    plt.ylabel(Labels_[0])
    plt.xlabel(Labels_[1])
    #plt.legend([GridPredictions], ["Training accuracy"])

    fig.savefig('result.png')
    


def SVCLearningCurve(DataTable, params, kernel_):
    acc_train, acc_val, numsamples = [], [], []
    N= np.shape(DataTable)[0]
    #for n in [20*x for x in range(N/30)]:
    for n in [50*x for x in range(N/50)]:
        if n==0:
            continue
        X,classifier= (DataTable[0:n-1,1:].astype(np.float), DataTable[0:n-1,0].astype(np.int))
        X_val,classifier_val=(DataTable[n:,1:].astype(np.float), DataTable[n:,0].astype(np.int))
        #print classifier
        if kernel_=='linear':
            TrainedModel=SVC(C=params[0], kernel=kernel_).fit(X,classifier)
        if kernel_=='rbf':
            TrainedModel=SVC(C=params[0], kernel=kernel_, gamma=params[1]).fit(X,classifier)
        acc_train.append(TrainedModel.score(X,classifier))
        acc_val.append(TrainedModel.score(X_val, classifier_val))

        numsamples.append(n)

    fig=plt.figure()
    ax=plt.gca()
    train=ax.scatter(numsamples,acc_train, color='red')
    crossval=ax.scatter(numsamples,acc_val, color='blue')
    #ax.set_yscale('log')
    plt.ylabel("Accuracy")
    plt.xlabel("Number of training samples")
    plt.legend([train,crossval], ["Training accuracy","Validation accuracy"], 'lower right')
    fig.savefig('learningCurve.png')



def SVCRegCurve(DataTable, Cvec,params_, kernel_,n):
    acc_train, acc_val = [], []
    for thisC in Cvec:
        X,classifier= (DataTable[0:n-1,1:].astype(np.float), DataTable[0:n-1,0].astype(np.int))
        X_val,classifier_val=(DataTable[n:,1:].astype(np.float), DataTable[n:,0].astype(np.int))
        if kernel_=='linear':
            TrainedModel=SVC(C=thisC, kernel=kernel_).fit(X,classifier)
        if kernel_=='rbf':
            TrainedModel=SVC(C=thisC, kernel=kernel_, gamma=params_[1]).fit(X,classifier)
        acc_train.append(TrainedModel.score(X,classifier))
        acc_val.append(TrainedModel.score(X_val, classifier_val))


    fig=plt.figure()
    ax=plt.gca()
    train=ax.scatter(Cvec,acc_train, color='red')
    crossval=ax.scatter(Cvec,acc_val, color='blue')
    #ax.set_yscale('log')
    plt.ylabel("Accuracy")
    plt.xlabel("Value of parameter C")
    plt.legend([train,crossval], ["Training accuracy","Validation accuracy"], 'lower right')
    fig.savefig('validationCurve.png')
    #fig.show()

def SVCGammaCurve(DataTable,Gammavec,params_, kernel_,n):
    acc_train, acc_val = [], []
    for thisGamma in Gammavec:
        X,classifier= (DataTable[0:n-1,1:].astype(np.float), DataTable[0:n-1,0].astype(np.int))
        X_val,classifier_val=(DataTable[n:,1:].astype(np.float), DataTable[n:,0].astype(np.int))
        if kernel_=='rbf':
            TrainedModel=SVC(C=params_[0], kernel=kernel_, gamma=thisGamma).fit(X,classifier)
        acc_train.append(TrainedModel.score(X,classifier))
        acc_val.append(TrainedModel.score(X_val, classifier_val))


    fig=plt.figure()
    ax=plt.gca()
    train=ax.scatter(Gammavec,acc_train, color='red')
    crossval=ax.scatter(Gammavec,acc_val, color='blue')
    #ax.set_yscale('log')
    plt.ylabel("Accuracy")
    plt.xlabel("Value of parameter gamma")
    plt.legend([train,crossval], ["Training accuracy","Validation accuracy"], 'lower right')
    fig.savefig('validationGammaCurve.png')
    #fig.show()


def SVCTwoDValidationCurve(DataTable,Gammavec,Cvec, kernel_,n):
    acc_train, acc_val, trainMinusValidation = [], [], []
    for thisGamma in Gammavec:
        for thisC in Cvec:
            X,classifier= (DataTable[0:n-1,1:].astype(np.float), DataTable[0:n-1,0].astype(np.int))
            X_val,classifier_val=(DataTable[n:,1:].astype(np.float), DataTable[n:,0].astype(np.int))
            if kernel_=='rbf':
                TrainedModel=SVC(C=thisC, kernel=kernel_, gamma=thisGamma).fit(X,classifier)
            #acc_train.append(TrainedModel.score(X,classifier))
            #acc_val.append(TrainedModel.score(X_val, classifier_val))
            trainMinusValidation.append(TrainedModel.score(X,classifier)-TrainedModel.score(X_val, classifier_val))


    X, Y = np.meshgrid(Cvec,Gammavec)
    #trainMinusValidation=np.array(acc_train)-np.array(acc_val)
    fig=plt.figure()
    #ax=plt.gca()
    #train=ax.scatter(Gammavec,acc_train, color='red')
    #crossval=ax.scatter(Gammavec,acc_val, color='blue')
    #ax.set_yscale('log')
    plt.pcolormesh(X,Y,np.array(trainMinusValidation))
    plt.colorbar() #need a colorbar to show the intensity scale
    plt.xlabel("Value of parameter C")
    plt.ylabel("Value of parameter gamma")
    #plt.legend([train,crossval], ["Training accuracy","Validation accuracy"])
    fig.savefig('validation2DCurve.png')
    #fig.show()

def main():
    args = getArguments()
    ## Class (0,1) should come first:
    checkFile(args.dataFile)
    Headers=np.genfromtxt(args.dataFile,delimiter=',',dtype=None)[0:1]
    ### Take the two X labels in order, and put the class last
    DataLabels=[Headers[0][1],Headers[0][2],Headers[0][0]]
    DataTable = np.genfromtxt(args.dataFile,delimiter=',',dtype=None)[1:]

    thisX=DataTable[:,1:].astype(np.float)
    if args.normalize:
        thisX=normalizeData(thisX)[0]
        DataTable[:,1:]=thisX
    if args.doPCA:
        print "do"
    
    print np.shape(DataTable)
    if args.logistic:
        if args.type=='linear':
            kernel='linear'
            C=1.0
            params=[C]
        elif args.type=='rbf':
            kernel='rbf'
            C=1.0
            gamma=.1
            params=[C,gamma]
        #kernel='rbf'

        if not args.numTrain:
            X,classifier= (DataTable[:,1:].astype(np.float), DataTable[:,0].astype(np.int))
            print "here"
            SVCLearningCurve(DataTable, params, kernel)
            print "here"
        else:
            N=int(args.numTrain)
            #Cvec=[0.1,1,2,3,4,5,10,20,30,40,50,60,70,80,90,100]
            #Cvec=[0.1+0.1*x for x in range(200)]
            Cvec=[1.+1.*x for x in range(20)]
            SVCRegCurve(DataTable,Cvec,params,kernel,N)
            if args.type=='rbf':
                #Gammavec=[.0001,.0002,.0003,.0004,.0005,.001,.01,]
                #Gammavec=[.0001*x+.0001 for x in range(200)]
                Gammavec=[.001*x+.001 for x in range(20)]
                SVCGammaCurve(DataTable,Gammavec,params, kernel,N)
                SVCTwoDValidationCurve(DataTable,Cvec,Gammavec, kernel,N)
            X,classifier= (DataTable[0:N-1,1:].astype(np.float), DataTable[0:N-1,0].astype(np.int))
            X_val,classifier_val=(DataTable[N:,1:].astype(np.float), DataTable[N:,0].astype(np.int))
            #TrainedModel=SVC(params, kernel = 'linear').fit(X,classifier)

            

        if kernel=='linear':
            TrainedModel=SVC(C=params[0], kernel = 'linear').fit(X,classifier)
        if kernel=='rbf':
            TrainedModel=SVC(C=params[0], kernel = 'rbf', gamma=params[1]).fit(X,classifier)
        ScatterWithLinearSVC(X,classifier,C,TrainedModel,DataLabels)
    
if __name__ == '__main__':
    main()
