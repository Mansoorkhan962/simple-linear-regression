import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error


data=pd.read_csv(r"C:\Users\mansoor khan\Desktop\machine_learning\marks_dataset.csv")

df=pd.DataFrame(data)

print(df.head())
print(df.columns)
print(df.shape)
print(df.describe())
print(df.info())

X=df[["Hours_Studied"]]
y=df["Exam_Marks"]

model=LinearRegression()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("rows available for training",len(X_train))
print("rows available for testing",len(X_test))

model.fit(X_train,y_train)

y_pred=model.predict(X_test)


mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)


print(f"MEAN ABSOLUTE ERROR : {round(mae,2)}")
print(f"MEAN SQUARED ERROR : {round(mse,2)}")
print(f"ROOT MEAN SQUARED VALUE : {round(rmse,2)}")
print(f"R2 SCORE : {round(r2,2)}")


comparison=pd.DataFrame({
        
                "ACTUAL_MARKS":y_test,
                "PREDICTED_MARKS":y_pred
 })

comparison["ERROR"]=(comparison['ACTUAL_MARKS'] - comparison["PREDICTED_MARKS"])
comparison['ABSOLUTE_ERROR']=comparison['ERROR'].abs()


print(comparison.head(10))


print(f"intercept : {round(model.intercept_,2)}")
print("coffient :",(model.coef_))



plot_data = pd.DataFrame({
    "Hours_Studied": X_test["Hours_Studied"].values,
    "Actual_Marks": y_test.values,
    "Predicted_Marks": y_pred
}).sort_values("Hours_Studied")

plt.figure(figsize=(10, 6))

# Actual test data points
plt.scatter(
    plot_data["Hours_Studied"],
    plot_data["Actual_Marks"],
    color="blue",
    alpha=0.7,
    label="Actual Marks (Test Data)"
)

# Predicted regression line
plt.plot(
    plot_data["Hours_Studied"],
    plot_data["Predicted_Marks"],
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Hours Studied")
plt.ylabel("Exam Marks")
plt.title("Simple Linear Regression: Hours Studied vs Exam Marks")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


plt.savefig("regression_plot.png", dpi=300, bbox_inches="tight")
plt.show()



new_student = pd.DataFrame({
    "Hours_Studied": [6]
})

predicted_marks = model.predict(new_student)

print("Hours Studied:", 6)
print("Predicted Exam Marks:", round(predicted_marks[0], 2))
