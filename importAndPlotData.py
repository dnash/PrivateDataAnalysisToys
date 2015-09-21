import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import argparse
import csv



def getArguments():
    parser = argparse.ArgumentParser(description='Import data from a csv file with headers on each column, and plot something...')

    # Command line flags
    parser.add_argument('-c', '--data_csv', action='store', dest='dataFile', help='The input csv data file')
    parser.add_argument('-p', '--plot_folder', action='store', dest='plotFolder', help='The output folder for the plots, must exist')
    parser.add_argument('-k', '--key_classifier', action='store', dest='classifier', help='The classifier to plot against for scatter plots')
    parser.add_argument('-n', '--number_train', action='store', dest='numTrain', help='The number of data points to use in training, the rest will be used for cross validation')
    #parser.add_argument('-r', '--clean_root', action='store_true', dest='doRoot', default=False, help='Preset cleaning of output root files/folders')

    args_ = parser.parse_args()
    return args_

def checkFile(file_):
    # Check that CSV file exists
    if not os.path.isfile(file_):
        print "Error: File %s does not exist." % file_
        print "Exiting with status 1."
        sys.exit(1)

def getData(csvfile_):
    reader= csv.DictReader(csvfile_, delimiter=',')
    Data={}
    for ind, row in enumerate(reader):
        keys=row.keys()
        if ind > 0:
            break
    for thiskey in row.keys():
        Data[thiskey]=[]

    for row in reader:
        for thiskey in keys:
            Data[thiskey].append(float(row[thiskey]))
    return Data

def Hist(data_,key_,nbins,folder_):
    thisdata=sorted(data_[key_],key=float)
    datarange=abs(thisdata[-1]-thisdata[0])
    binning=[]
    for i in range(nbins+1):
        binning.append(thisdata[0]+i*(datarange/nbins))
    #thisdata[:] = [x / float(len(thisdata)) for x in thisdata]
    plt.hist(thisdata,bins=binning,normed=True)
    plt.ylabel('Fraction of data sample')
    plt.xlabel(key_)
    plt.savefig(folder_+'/histo_'+key_.replace(' ','')+'.png')
    plt.clf()


def Scatter(data_,key1_, key2_,folder_):
    thisdata1=sorted(data_[key1_],key=float)
    thisdata2=sorted(data_[key2_],key=float)

    plt.scatter(thisdata1, thisdata2)
    plt.ylabel(key2_)
    plt.xlabel(key1_)
    plt.savefig(folder_+'/scatter_'+key1_.replace(' ','')+'_vs_'+key2_.replace(' ','')+'.png')
    plt.clf()

def linearRegCost(X,y,theta,reg_lambda):
    m=float(np.shape(y)[0])
    theta=np.reshape(theta,(np.shape(theta)[0],1))   ###It seems nRows is sometimes lost when passing through fmin_cg, reshape to be right here...
    Pulls=X*theta-y
    PullsSquaredAndSummed = Pulls.transpose()*Pulls
    RegTerm=reg_lambda/(2*m) * np.dot(theta[1:].transpose(),theta[1:])
    Cost = 1/(2*m) * PullsSquaredAndSummed + RegTerm
   
    theta_for_grad_reg = theta
    theta_for_grad_reg[0]=0

    grad= (1/m) * (X.transpose() * Pulls)  + (reg_lambda/m)*theta_for_grad_reg


    return (Cost,grad)

def makeDataMatrices(data,classifier_key):
    y= np.matrix(data[classifier_key])
    listofinputarrays=[]
    for thiskey in data.keys():
        if thiskey !=classifier_key:
            listofinputarrays.append(np.array(data[thiskey]))
    X=np.c_[listofinputarrays]
    X=np.transpose(X)
    ## Adding a column of ones to X
    onescolumn=np.transpose(np.matrix(np.ones(np.shape(X)[0])))
    X=np.concatenate((onescolumn,X),axis=1)
    X=np.matrix(X)
    return [X,y.transpose()]

def makeAndSplitDataMatrices(data,classifier_key,numTrain):
    [X,y] = makeDataMatrices(data,classifier_key)
    Xtrain=X[0:int(numTrain),:]
    ytrain=y[0:int(numTrain),:]
    Xcrossval=X[int(numTrain)+1:np.shape(X)[0],:]
    ycrossval=y[int(numTrain)+1:np.shape(y)[0],:]
    return Xtrain,ytrain,Xcrossval,ycrossval
    
            

def trainLinearReg(X,y,reg_lambda):
    theta_start = np.asarray(np.zeros((np.shape(X)[1]),float))
    args=[X,y,reg_lambda]
    def lRegCost(theta, *args):
        X, y, reg_lambda = args
        [Cost,grad]=linearRegCost(X,y,theta, reg_lambda)
        return float(Cost)
    def lRegGrad(theta, *args):
        X, y, reg_lambda = args
        [Cost,grad]=linearRegCost(X,y,theta, reg_lambda)
        #grad=np.reshape(grad, np.shape(grad)[0] )
        grad = np.ravel(grad)
        #print np.shape(np.reshape(grad,np.shape(grad)[0]))
        return np.asarray((grad))
    theta = opt.fmin_cg(lRegCost,theta_start,fprime=lRegGrad,args=args,maxiter=200,disp=False)
    return theta

def learningCurve(X,y,X_crossval,y_crossval,reg_lambda):
    N = np.shape(X)[0]
    err_train=[]
    err_crossval=[]
    numsamples=[]
    j=0
    for i in [20*x for x in range(N/20)]:
        if i==0:
            continue
        theta=trainLinearReg(X[0:i,:],y[0:i],reg_lambda)
        err_train.append(linearRegCost(X[0:i,:],y[0:i],theta,0)[0])
        err_crossval.append(linearRegCost(X_crossval,y_crossval,theta,0)[0])
        numsamples.append(i)
        print "On iteration"+str(j)
        print "err_train = "+str(err_train[j])
        print "err_crossval = "+str(err_crossval[j])
        j+=1

    fig=plt.figure()
    ax=plt.gca()
    ax.scatter(numsamples,err_train)
    ax.scatter(numsamples,err_crossval, color='blue')
    ax.set_yscale('log')
    ax.ylabel("Error")
    ax.xlabel("Number of training samples")
    fig.savefig('test.png')
    fig.show()

def featureNormalize(X):
    N=np.shape(X)[1]
    X_norm=X
    for i in range(N):
        mu=np.mean(X[:,i])
        sigma=np.std(X[:,i])
        X_norm[:,i]=(X_norm[:,i]-mu)/sigma
    return X_norm
    

def main():
    args = getArguments()
    checkFile(args.dataFile)
    csvfile=open(args.dataFile,'r')
    data=getData(csvfile)
    print "Data available:"
    print data.keys()
    nbins=10
    if args.plotFolder:
        for thiskey in data.keys():
            Hist(data,thiskey,nbins,args.plotFolder)
        if args.classifier:
            for thiskey in data.keys():
                if thiskey != args.classifier:
                    Scatter(data,args.classifier,thiskey,args.plotFolder)
    if args.classifier:
        if not args.numTrain:
            [Data_matrix, result_vector] = makeDataMatrices(data,args.classifier)
        elif args.numTrain:
            [Data_matrix, result_vector, CrossVal_matrix, CrossVal_result] = makeAndSplitDataMatrices(data,args.classifier,args.numTrain)
            Data_matrix[:,1:]=featureNormalize(Data_matrix[:,1:])
            learningCurve(Data_matrix, result_vector, CrossVal_matrix, CrossVal_result,0)
        #theta=trainLinearReg(Data_matrix,result_vector,0)
        #print theta
        

    #for thiskey in data.keys():
    #    print data[thiskey]
    
    
 

if __name__ == '__main__':
    main()
