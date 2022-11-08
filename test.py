import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def test(data):
    data.drop(columns=['StandardHours', 'EmployeeCount', 'Over18','EmployeeNumber'],axis=1, inplace=True)
    data.Education = data.Education.astype('category')
    data.EnvironmentSatisfaction = data.EnvironmentSatisfaction.astype('category')
    data.JobInvolvement = data.JobInvolvement.astype('category')
    data.JobSatisfaction = data.JobSatisfaction.astype('category')
    data.PerformanceRating = data.PerformanceRating.astype('category')
    data.RelationshipSatisfaction = data.RelationshipSatisfaction.astype('category')
    data.WorkLifeBalance = data.WorkLifeBalance.astype('category')
    data.BusinessTravel = data.BusinessTravel.astype('category')
    data.Department = data.Department.astype('category')
    data.EducationField = data.EducationField.astype('category')
    data.Gender = data.Gender.astype('category')
    data.JobRole = data.JobRole.astype('category')
    data.MaritalStatus = data.MaritalStatus.astype('category')
    data.OverTime = data.OverTime.astype('category')
    for col in data.columns:
        if data[col].dtype == 'category' or data[col].dtype == 'object':
            data = pd.get_dummies(data, columns=[col], prefix = [col],drop_first=True)
       
    data.rename(columns={'Attrition_Yes': 'Attrition'}, inplace=True)
    data.Attrition = data.Attrition.astype('int64')
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data

       