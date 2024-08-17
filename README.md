{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score,classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv(\"emails.csv\")\n",
    "x=data['text']\n",
    "y=data['label']\n",
    "vectorizer=CountVectorizer()\n",
    "x_counts=vectorizer.fit_transform(x)\n",
    "tfidf_transformer=TfidfTransformer()\n",
    "x_tfidf=tfidf_transformer.fit_transform(x_counts)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test=train_test_split(x_tfidf,y,test_size=0.2,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "nb_model=MultinomialNB()\n",
    "nb_model.fit(X_train,y_train)\n",
    "y_pred_nb=nb_model.predict(X_test)\n",
    "print(\"Naive Bayes Classification Report\")\n",
    "print(classification_report(y_test,y_pred_nb))\n",
    "print(\"Accuracy\",accuracy_score(y_test,y_pred_nb))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "svm_model=SVC(kernel='linear')\n",
    "svm_model.fit(X_train,y_train)\n",
    "y_pred_svm=svm_model.predict(X_test)\n",
    "print(\"SVM Classification Report\")\n",
    "print(classification_report(y_test,y_pred_svm))\n",
    "print(\"Accuracy\",accuracy_score(y_test,y_pred_svm))"
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
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
