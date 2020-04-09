using CSV
using Plots
using Statistics
using LinearAlgebra
using DataFrames

dataset = CSV.read("sample_folder/data/housingPriceData.csv")


# function for partitioning the training and testing dataset

# an important note : this will change the training data each time it runs because it shuffles the dataset


using Random
df=dataset
sample = randsubseq(1:size(df,1), 0.8)
train = df[sample, :]
notsample = [i for i in 1:size(df,1) if isempty(searchsorted(sample, i))]
test = df[notsample, :]

x1=train.bedrooms
x2=train.bathrooms
x3=train.sqft_living

m = length(x1)
x0 = ones(m)
x0_mean=mean(x0)
x0_sd=std(x0)

x1_mean=mean(x1)
x1_sd=std(x1)

x2_mean=mean(x2)
x2_sd=std(x2)

x3_mean=mean(x3)
x3_sd=std(x3)


y=train.price

y_mean=mean(y)
y_sd=std(y)


function feature_scale(x)
    x_mean=mean(x)
    x_sd=std(x)
    x=(x.-x_mean)/x_sd
    return x
end


x1=feature_scale(x1)
x2=feature_scale(x2)
x3=feature_scale(x3)

X = cat(x0, x1, x2,x3, dims=2)

# here comes the ridge function
function ridge(X,y,k)
    temp=X'*X
    kI=k*Matrix{Float64}(I, 4, 4)
    left_mat=temp+kI
    Betas=inv(left_mat)*(X'*y)
    return Betas
end


Betas=ridge(X,y,0.0004)

# Rmse function
function rmse(y,y_hat)
    n=length(y)
    temp=y_hat-y
    r=sum(temp.^2)/(2*n)
    return sqrt(r)
end


# R squared function
function r_2(y,y_hat)
    n=length(y)
    temp=y_hat-y
    num=sum(temp.^2)
    y_bar=mean(y)
    den=sum(y.^2)-(n*y_bar^2)
    return 1-(num/den)
end

ytrain_pred=X*Betas

# here is the code for evaluation
x1_test=test.bedrooms
x2_test=test.bathrooms
x3_test=test.sqft_living

m_test = length(x1_test)
x0_test = ones(m_test)

x1_test=feature_scale(x1_test)
x2_test=feature_scale(x2_test)
x3_test=feature_scale(x3_test)

X_test = cat(x0_test, x1_test, x2_test,x3_test, dims=2)
y_pred=X_test*Betas
y_test=test.price
rms=rmse(y_test,y_pred)
r_s=r_2(y_test,y_pred)

df=DataFrame(Price=y_pred)
CSV.write("data/2b.csv",df) # I am dumping only the predicted values for test data
