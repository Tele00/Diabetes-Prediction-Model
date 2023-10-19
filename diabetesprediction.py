import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
from tkinter import *
import threading

result = ""
screen = Tk()
screen.geometry("600x600")
screen.title("Diabetes Prediction System")

def predict():
    dataset = pd.read_csv('diabetes.csv')
    filename = 'trained_diabetes_model.sav'

    X = dataset.drop(columns = 'Outcome', axis=1)
    Y = dataset['Outcome']

    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    print(standardized_data)

    X = standardized_data
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

    classifier = svm.SVC(kernel='linear')


    # training the SVM Classifier
    classifier.fit(X_train, Y_train)
    filename = 'trained_diabetes_model.sav'
    pickle.dump(classifier, open(filename, 'wb'))



    # accuracy score on training data
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy Score of training data : ', training_data_accuracy)

    # accuracy score on test data
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy Score of test data : ', test_data_accuracy)



    loaded_classifier = pickle.load(open(filename, 'rb'))

    input_data = (preg.get(), gc.get(), bp.get(), st.get() , insulin_level.get() , bodymass.get() , dp.get(), age.get())

    # changing the input data to a numpy array
    numpy_array = np.asarray(input_data)

    # reshape the array because we are predicting for one instance
    reshaped_input_data = numpy_array.reshape(1, -1)

    # standardize the input data
    std_data = scaler.transform(reshaped_input_data)

    # make the prediction
    prediction = loaded_classifier.predict(std_data)

    if (prediction[0] == 0):
        result = "This person is not diabetic"
        result_label = Label(text = result, fg = "green", width = "30")
        result_label.place(x = 140, y = 480)
        print("This person is not diabetic")
    else:
        result = "This person is diabetic"
        result_label = Label(text = result, fg = "red", width = "30")
        result_label.place(x = 140, y = 480)
        screen.update()
        print("This person is diabetic")



heading = Label(text="Predict If A Patient Has Diabetes", bg="white", fg="black")
heading.pack()


pregnancy = Label(text="Pregnancy *")
glucose = Label(text="Glucose *")
blood_pressure = Label(text="Blood Pressure *")
skin_thickness = Label(text="Skin Thickness *")
insulin = Label(text="Insulin Level *")
bmi = Label(text="BMI *")
diabetes_pedigree = Label(text="Diabetes Pedigree Function *")
age = Label(text="Age *")
pregnancy.place(x = 15, y = 50)
glucose.place(x = 15, y = 100)
blood_pressure.place(x = 15, y = 150)
skin_thickness.place(x = 15, y = 200)
insulin.place(x = 15, y = 250)
bmi.place(x = 15, y = 300)
diabetes_pedigree.place(x = 15, y = 350)
age.place(x = 15, y = 400)

preg = DoubleVar()
gc = DoubleVar()
bp = DoubleVar()
st = DoubleVar()
insulin_level = DoubleVar()
bodymass = DoubleVar()
dp = DoubleVar()
age = IntVar()

preg_entry = Entry(textvariable = preg)
gc_entry = Entry(textvariable = gc)
bp_entry = Entry(textvariable = bp)
st_entry = Entry(textvariable = st)
insulin_level_entry = Entry(textvariable = insulin_level)
bodymass_entry = Entry(textvariable = bodymass)
dp_entry = Entry(textvariable = dp)
age_entry = Entry(textvariable = age)

preg_entry.place(x = 15, y = 75)
gc_entry.place(x = 15, y = 125)
bp_entry.place(x = 15, y = 175)
st_entry.place(x = 15, y = 225)
insulin_level_entry.place(x = 15, y = 275)
bodymass_entry.place(x = 15, y = 325)
dp_entry.place(x = 15, y = 375)
age_entry.place(x = 15, y = 425)

result_label = Label(text = result)
result_label.place(x = 150, y = 480)

predict = Button(text = "Predict", width="30", height="2", command = predict)
predict.place(x = 150, y = 520)



screen.mainloop()
