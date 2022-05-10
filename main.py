import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.feature_selection import mutual_info_regression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
import os

# PAGE LAYOUT AND SETUP
st.set_page_config(page_title="AUTOMATION", page_icon=":robot_face:", layout="wide")
st.title(":robot_face: Dataset processing automation")
st.write('This could only work on a certain type of datasets, and the dataset MUST NOT EXCEED 200MB')


#FILE UPLOAD AND FIRST FILE MANIPULATION

file = st.file_uploader("Choose a file")
file2 = st.file_uploader("Upload a Machine learning Model (Optional)")
exten = st.selectbox(
"File extension: ",
('csv', 'xls' ,'xlsx', 'json', 'sql'))
nas = []

#CSV FILE
if exten == 'csv':
    st.write('Dataframe settings')

    #   DEFAULT SETTINGS READ_CSV
    if exten == 'csv':
        read_settings = st.checkbox('CSV file settings')
        sep = ','
        delimiter = None
        header = 'infer'
        index_co = 0

    #      CSV  FILE  READ SETTINGS

    if read_settings:
        sep = st.text_input('Seperator', value=',')
        delimiter = st.text_input('Delimiter', value=None)
        if delimiter == 'None':
            delimiter = None
        header = st.text_input('Header', value='infer')
        index_co = st.number_input('Index column by number', value=0)
        new_index = st.checkbox('Add new default index instead')
        if new_index or (not read_settings):
            index_co = None
        st.write('******************************************')

    #read = st.button('Read the file')
    read = st.checkbox('Read the CSV: ')
    if read:
        df = pd.read_csv(file, sep = sep, delimiter = delimiter, header = header, index_col = index_co)
        st.header('The dataframe: ')
        st.dataframe(df)

        #THE DOWNLOAD AS CSV OPTION
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)
        st.download_button('Download dataframe as CSV',csv, "file.csv", "text/csv")
        st.header('DataFrame description: ')
        df_columns = df.columns.tolist()
        desc = df.describe()

        #Trying to show the dtypes table as a dataframe
        type_list = []
        types = df.dtypes

        xd = []
        xd1 = []
        xd2 = []
        for i in df.columns:
            Column_type = df[i].dtypes.name
            Nan_count = df[i].isna().sum()
            unique_count = len(df[i].unique())
            mum = {'Column name': i, 'Column type': Column_type, 'Nan count' : Nan_count, 'unique values count' : unique_count}
            xd.append(mum)

# GENERAL INFORMATION ABOUT DATAFRAME
        st.write(f"DataFrame shape: {df.shape}")
        xd = pd.DataFrame.from_dict(xd)  # Basically converting column types to strings so I can use them later
        left_column,right_column = st.columns(2)
        left_column.dataframe(desc)
        right_column.dataframe(xd)


        for i in df_columns:
           na_perc = (df[i].isna().sum() / len(df[i])) * 100
           nas.append(na_perc)
        st.header('Missing Values distribution: ')
        nas1 = { 'Column Name' : df_columns , 'NaN %' : nas }
        fig1 = px.bar(nas1, x='Column Name', y='NaN %' , color='NaN %' , title='Nan ratios by columns')
        fig2, ax = plt.subplots()
        sns.heatmap(df.isna(), ax= ax)
        left_column, mid, right_column = st.columns(3)
        left_column.plotly_chart(fig1)
        right_column.pyplot(fig2)

        #MORE CLEANING AND FILE MANIPULATION
        st.header('Dealing with missing values: ')
        st.write('LEAVE NO NAN after this step')
        X = df.copy()
        st.write('Mean :Nans replaced by Mean for numerals and most common value for categorical or boolean ')
        st.write(' Zero : Nans replaced by zero for numerals and Unknown for categoricals and False for booleans ')
        st.write('Delete: Delete columns')
        left, mid, right = st.columns(3)
        Nan_mean = left.multiselect('Replace columns by Mean :  ', df_columns)
        apply_mean = left.checkbox('Apply Mean')
        Nan_zero = mid.multiselect('Replace columns by Zero: ', df_columns)
        apply_zero = mid.checkbox('Apply zero')
        Nan_del = right.multiselect('Delete Columns: ', df_columns)
        apply_del = right.checkbox('Apply Delete')
        left,right = st.columns(2)
        if (Nan_mean and apply_mean):
            for i in Nan_mean: # Filling NaNs by Mean
                if X[i].dtypes.name == ("float64" or "int64"):
                    X[i] = X[i].fillna(X[i].mean())
                elif X[i].dtypes.name == "object":
                    com = X[i].value_counts().index[0]
                    X[i] = X[i].fillna(com)
                elif X[i].dtypes.name == "bool":
                    com = X[i].value_counts().index[0]
                    X[i] = X[i].fillna(com)
                else:
                    right.write('wf is that type bruh')
        if (Nan_zero and apply_zero): # Filling NaNs by ZEROS
            for i in Nan_zero:
                if X[i].dtypes.name == ("float64" or "int64"):
                     X[i] = X[i].fillna(0)
                elif X[i].dtypes.name == "object":
                    X[i] = X[i].fillna('Unknown')
                elif X[i].dtypes.name == "bool":
                    X[i] = X[i].fillna(False)
                else:
                    right.write('wtf is that column type bruh')
        if (Nan_del and apply_del):
            X = X.drop(axis=1, columns=Nan_del)

        for i in X.columns:
            Nan_count = X[i].isna().sum()
            stuff1 = {'Column name': i, 'Nan count': Nan_count}
            xd1.append(stuff1)
        left.dataframe(X)
        right.dataframe(xd1)
        X_columns = X.columns.tolist()

    # MORE PREPROCESSING
        st.header('More preprocessing')
        st.write('Do not use One-hot encoder on features with a high unique value count! Use factorization instead')
        st.write('There should be NO feature type OBJECT left before preprocessing')
        left, right = st.columns(2)

        fact = left.multiselect('Columns to factorize :  ', X_columns)
        apply_fact = left.checkbox('Apply factorization')
        one_hot = right.multiselect('Columns for One-hot-encoder: ', X_columns)
        apply_onehot = right.checkbox('Apply One hot encoders')

        # ONE HOT ENCODER
        if (one_hot and apply_onehot):
            X_dum = pd.get_dummies(X[one_hot])
            #X = pd.merge(left=X, right=X_dum, how='cross')
            #X = X.drop(axis=1, columns=one_hot)
            X = pd.merge(X, X_dum, left_index=True, right_index=True)
            X = X.drop(axis=1, columns=one_hot)





        #FACTORIZATION
        if (fact and apply_fact):
              # removes the target column from the DF and returns it to y
            for colname in X[fact].select_dtypes("object"):
                X.select_dtypes("object")
                X[colname], _ = X[colname].factorize()  # factorize turns distinct values to int
                # All discrete features should now have integer dtypes (double-check this before using MI!)

            discrete_features = X.dtypes.name == ("int64" or "int" or "int32" or "bool")

        # Visualization

        for i in X.columns:
            Column_type = X[i].dtypes.name
            Nan_count = X[i].isna().sum()
            unique_count = len(X[i].unique())
            stuff1 = {'Column name': i, 'Column type': Column_type, 'Nan count': Nan_count,
                            'unique values count': unique_count}
            xd2.append(stuff1)

        left, right = st.columns(2)
        xd2 = pd.DataFrame.from_dict(xd2)
        left.dataframe(X)
        right.dataframe(xd2)
        processed_csv = convert_df(X)
        st.download_button('Download dataframe as CSV', processed_csv, "processed_file.csv", "text/csv")

       # MACHINE LEARNING ALGORITHMS

        st.header('Machine learning : ')

        #MUTUAL INFORMATION REGRESSION
        mi_reg = st.checkbox('Apply Mutual Information regression')
        if mi_reg:
            l, m, r = st.columns(3)
            target = m.selectbox('Select Target feature: ', X_columns)
            apply_target = m.checkbox('Apply target selection')
            m.write('******************************')
            if (target and apply_target):
                y = X.pop(target)
                discrete_features = X.dtypes.name == ("int64" or "int" or "int32" or "bool")
                mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
                mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
                mi_scores = mi_scores.sort_values(ascending=False)
                mi_df = mi_scores.to_frame(name='MI_Scores')
                left, right = st.columns(2)

                right.write(mi_scores)  # show a few features with their MI scores
                # Plotting MI SCORES
                fig3 = go.Figure(go.Bar(x=mi_df.MI_Scores, y=mi_df.index, orientation='h'))
                left.plotly_chart(fig3)


        # ALREADY HAVE A MODEL
        prediction = st.checkbox('Already uploaded a model? Predict now')
        l,m,r = st.columns(3)

        if prediction:
            pred = m.text_input('Prediction target feature name:  ')
            conf = m.checkbox('Apply model ')
            m.write('*********************************************************************')
            if pred and conf:
                loaded_model = pickle.load(file2)
                X[pred] = loaded_model.predict(X)
                st.write('Your predicted dataframe: ')
                st.dataframe(X)
                csv2 = X.to_csv().encode('utf-8')
                st.download_button('Download prediction dataframe as CSV', csv2, "file.csv", "text/csv")


        classif = st.checkbox('Classification and Linear Regression')
        if classif:
            target = st.selectbox('Select Target feature for classification: ', X_columns)
            apply_classif = st.checkbox('Apply classification')
            y = X.pop(target)
        

        if (classif and target and apply_classif):
            st.write('This might take a while...')
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


          # TESTING MODELS
            gnb = GaussianNB()
            gnb_test = gnb.fit(x_train, y_train)
            acc_gnb = gnb.score(x_test, y_test)

            #dt = tree.DecisionTreeClassifier(random_state=1)
            #dt_test = dt.fit(x_train, y_train)
            #acc_dt = dt.score(x_test, y_test)

            knn = KNeighborsClassifier()
            knn_test = knn.fit(x_train, y_train)
            acc_knn = knn.score(x_test, y_test)

            rf = RandomForestClassifier(random_state=1)
            rf_test = rf.fit(x_train, y_train)
            acc_rf = rf.score(x_test, y_test)

            lr = LogisticRegression()
            lr_test = lr.fit(x_train, y_train)
            acc_lr = lr.score(x_test, y_test)


            svc = SVC(probability = True)
            svc_test = svc.fit(x_train, y_train)
            acc_svc = svc.score(x_test, y_test)

            lin = LinearRegression()
            lin_test = lin.fit(x_train, y_train)
            acc_lin = lin.score(x_test, y_test)

            m = {'Model': ['Gaussian NB', 'Logistic Regression', 'Random Forest', 'KNN', 'SVM', 'Linear Regression'] ,
                 'Accuracy': [acc_gnb, acc_lr, acc_rf, acc_knn, acc_svc, acc_lin]}
            models = pd.DataFrame(data=m)
            models = models.sort_values(by = ['Accuracy'], ascending=False)
            models = models.reset_index(drop=True)
            st.dataframe(models)

            #DOWNLOAD THE MODEL

            download = st.checkbox('Download best model')
            if download:

                if models['Model'][0] == 'Gaussian NB':
                    filename = 'GaussianNb_model.sav'
                    pickle.dump(gnb_test, open(filename, 'wb'))
                if models['Model'][0] == 'Logistic Regression':
                    filename = 'Logistic_Regression.sav'
                    pickle.dump(lr_test, open(filename, 'wb'))
                if models['Model'][0] == 'KNN':
                    filename = 'KNN_model.sav'
                    pickle.dump(knn_test, open(filename, 'wb'))
                if models['Model'][0] == 'Random Forest':
                    filename = 'Random_Forest_model.sav'
                    pickle.dump(rf_test, open(filename, 'wb'))
                if models['Model'][0] == 'SVM':
                    filename = 'SVM_model.sav'
                    pickle.dump(svc_test, open(filename, 'wb'))
                if models['Model'][0] == 'Linear Regression':
                    filename = 'Linear_Regression_model.sav'
                    pickle.dump(lin_test, open(filename, 'wb'))

                #if models['Model'][0] == 'Decision Tree':
                    #filename = 'finalized_model.sav'
                    #pickle.dump(dt_test, open(filename, 'wb'))

                st.write('Model saved!')
                st.write('If you want to use the model for prediction, rerun the whole app and upload it')









