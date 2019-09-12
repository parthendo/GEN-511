{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn import svm\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn import preprocessing\n",
    "from sklearn.gaussian_process.kernels import RBF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"Pima_Indian_diabetes.csv\")\n",
    "category = [\"Pregnancies\",\"Glucose\",\"BloodPressure\",\"SkinThickness\",\"Insulin\",\"BMI\",\"DiabetesedigreeFunction\",\"Age\",\"Outcome\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Removing all the rows having any of the null values or numeric value 0\n",
    "\n",
    "# As non-diabetic class had value 0, replace it with 2 to reduce confusion\n",
    "data[\"Outcome\"] = data[\"Outcome\"].replace(to_replace = 0, value = 2)  \n",
    "# Replace all the cells having 0 with NaN\n",
    "data = data.replace(to_replace = 0, value = np.nan) \n",
    "#data.isnull().sum()\n",
    "# Drop rows having any null value\n",
    "data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#NTS: Plot the new data on the graph ****************\n",
    "# Training a model after removing null values\n",
    "values = data.values\n",
    "# Data\n",
    "X = values[:,0:8]\n",
    "# Labels\n",
    "Y = values[:,8]\n",
    "# Splitting data in test and training set\n",
    "X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=0)\n",
    "X_train = preprocessing.scale(X_train)\n",
    "X_test = preprocessing.scale(X_test)\n",
    "# Training the SVM model\n",
    "sVm = svm.SVC(gamma=2,C=0.01,decision_function_shape='ovo',kernel='rbf')\n",
    "sVm.fit(X_train,Y_train)\n",
    "pred = sVm.predict(X_test)\n",
    "count = 0\n",
    "for i in range(len(Y_test)):\n",
    "    if(Y_test[i] == pred[i]):\n",
    "        count = count + 1\n",
    "print(float(count)/len(Y_test)*100)\n",
    "\n",
    "# Accuracy acheived after eliminating null value rows is: 67.41573033707866%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Replace null value with mean a.k.a imputation\n",
    "# 1. Fill with mean\n",
    "# 2. Take the minimum and maximum of the range and fill with a random value in that range\n",
    "\n",
    "# Re-initializing dataframe with replacing 0s with NaN\n",
    "data = pd.read_csv(\"Pima_Indian_diabetes.csv\")\n",
    "category = [\"Pregnancies\",\"Glucose\",\"BloodPressure\",\"SkinThickness\",\"Insulin\",\"BMI\",\"DiabetesedigreeFunction\",\"Age\",\"Outcome\"]\n",
    "data[\"Outcome\"] = data[\"Outcome\"].replace(to_replace = 0, value = 2)  \n",
    "data = data.replace(to_replace = 0, value = np.nan) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "ename": "IndexError",
     "evalue": "list assignment index out of range",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mIndexError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-29-60c953d139b2>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m8\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 6\u001b[0;31m     \u001b[0mmn\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mcategory\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      7\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmn\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mIndexError\u001b[0m: list assignment index out of range"
     ]
    }
   ],
   "source": [
    "# 1. Replace NaN with mean\n",
    "\n",
    "mn = []\n",
    "\n",
    "for i in range(0,7):\n",
    "    mn[i] = data[category[i]].mean()\n",
    "print(mn)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
