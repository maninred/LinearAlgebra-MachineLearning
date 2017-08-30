from __future__ import division
import xlrd as xl
import numpy as np
# import matplotlib.pyplot as mapy

workbook=xl.open_workbook("university data.xlsx")
sheet=workbook.sheet_by_index(0)

CS_array1=np.array([[]])
Research_array1=np.array([[]])
Admin_array1=np.array([[]])
Tuition_array1=np.array([[]])

for n in range(1, 50):
    b = np.array([[sheet.cell_value(n,2)]])
    CS_array1 = np.concatenate((CS_array1,b), axis=1)
    c = np.array([[sheet.cell_value(n,3)]])
    Research_array1 = np.concatenate((Research_array1, c), axis=1)
    d = np.array([[sheet.cell_value(n,4)]])
    Admin_array1 = np.concatenate((Admin_array1, d), axis=1)
    e = np.array([[sheet.cell_value(n,5)]])
    Tuition_array1 = np.concatenate((Tuition_array1, e), axis=1)

CS_array = [i for i in CS_array1[0]]
Research_array = [i for i in Research_array1[0]]
Admin_array = [i for i in Admin_array1[0]]
Tuition_array = [i for i in Tuition_array1[0]]

mu1=np.around(np.mean(CS_array),3)
mu2=np.around(np.mean(Research_array),3)
mu3=np.around(np.mean(Admin_array),3)
mu4=np.around(np.mean(Tuition_array),3)

var1=np.around(np.var(CS_array),3)
var2=np.around(np.var(Research_array),3)
var3=np.around(np.var(Admin_array),3)
var4=np.around(np.var(Tuition_array),3)

sigma1=np.around(np.std(CS_array),3)
sigma2=np.around(np.std(Research_array),3)
sigma3=np.around(np.std(Admin_array),3)
sigma4=np.around(np.std(Tuition_array),3)

print 'UBitName = mjarugu'
print 'personNumber = 50206843'
print 'mu1 = '+str(mu1)
print 'mu2 = '+str(mu2)
print 'mu3 = '+str(mu3)
print 'mu4 = '+str(mu4)
print 'var1 = '+str(var1)
print 'var2 = '+str(var2)
print 'var3 = '+str(var3)
print 'var4 = '+str(var4)
print 'sigma1 = '+str(sigma1)
print 'sigma2 = '+str(sigma2)
print 'sigma3 = '+str(sigma3)
print 'sigma4 = '+str(sigma4)


X = np.vstack((CS_array,Research_array,Admin_array,Tuition_array))
XName=('CS_array','Research_array','Admin_array','Tuition_array')
covarianceMat=np.cov(X)
correlationMat=np.corrcoef(X)

# graphsno = 1
#
# for i in range(0,3):
#     for j in range(i+1,4):
#         mapy.figure(graphsno)
#         mapy.scatter(X[i],X[j],s=50)
#         mapy.xlabel(XName[i])
#         mapy.ylabel(XName[j])
#         mapy.title(XName[i] + " Vs " + XName[j])
#         graphsno += 1
# mapy.show()

print 'covarianceMat ='
print np.round(covarianceMat,3)
print 'correlationMat ='
print np.around(correlationMat,3)

formaxmin=np.array(correlationMat)
for x, y in np.ndindex(formaxmin.shape):
    if not x > y:
        formaxmin[x][y] = None
    else:
        formaxmin[x][y] = np.abs(formaxmin[x][y])

maxi=np.nanmax(formaxmin)
mini=np.nanmin(formaxmin)

for x, y in np.ndindex(formaxmin.shape):
    if formaxmin[x][y]==maxi:
        print "Maximum correlation between "+XName[x]+" and "+XName[y]
    elif formaxmin[x][y] == mini:
        print "Minimum correlation between " + XName[x] + " and " + XName[y]

def loghood(a1,mu,sigma):
    a1=((1/(abs(sigma)*np.sqrt(2*np.pi)))*(np.exp((-0.5)*((a1-mu)/abs(sigma))**2)))
    a1 = np.log(a1)
    return a1


P=np.vstack((loghood(CS_array,mu1,sigma1),loghood(Research_array,mu2,sigma2),loghood(Admin_array,mu3,sigma3),loghood(Tuition_array,mu4,sigma4)))
logLikelihood=np.around(np.sum(P),3)
print 'logLikelihood ='
print logLikelihood

allrow=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
for x,y in np.ndindex(allrow.shape):
    if not x>y:
        allrow[x][y]=0
    else:
        allrow[x][y]=1
BNgraph=allrow
print 'BNgraph ='
print BNgraph


def linearloglikelyhood(one,two):
    N=len(one)
    two.insert(0,[1]*N)
    K = len(two)

    A=[]
    for i in range(K):
        row=[]
        for j in range(K):
            ele=0
            for ik in range(N):
                ele+=two[i][ik]*two[j][ik]
            row.append(ele)
        A.append(row)

    B=[]
    for j in range(K):
        ele=0
        for ik in range(N):
            ele+=one[ik]*two[j][ik]
        B.append([ele])

    Beta=np.linalg.solve(A,B)

    finalsum=0
    for i in range(N):
        sums=0
        for j in range(K):
            sums+=Beta[j]*two[j][i]
        finalsum+=(sums - one[i])**2
    sigs=finalsum.item(0)/N

    logfinal=(-0.5)*N*(np.log(2*np.pi*sigs)+1)
    return logfinal

BNlogLikelihood=np.around(linearloglikelyhood(CS_array,[Research_array,Admin_array,Tuition_array])+linearloglikelyhood(Research_array,[Admin_array,Tuition_array])+linearloglikelyhood(Admin_array,[Tuition_array])+linearloglikelyhood(Tuition_array,[]),3)
print "BNlogLikelihood ="
print BNlogLikelihood

# BNlogLikelihood1=np.around(linearloglikelyhood(CS_array,[])+linearloglikelyhood(Research_array,[])+linearloglikelyhood(Admin_array,[])+linearloglikelyhood(Tuition_array,[]),3)
# print "The unconnected BN log likelyhood "
# print BNlogLikelihood1
#
# BNlogLikelihood2=np.around(linearloglikelyhood(CS_array,[Research_array,Tuition_array])+linearloglikelyhood(Research_array,[Admin_array])+linearloglikelyhood(Admin_array,[Tuition_array])+linearloglikelyhood(Tuition_array,[]),3)
# print "The unconnected BN log likelyhood "
# print BNlogLikelihood2