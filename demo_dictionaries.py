"""
Demo data dictionaries that match the demo datasets in web_app.py
These are used for testing the data dictionary parser functionality
"""

DEMO_DICTIONARIES = {
    "CSV - Western": """Column,Type,Required,Min,Max,Description,Allowed_Values
employee_id,integer,Yes,1000,9999,Employee ID (4 digits),
first_name,string,Yes,,,First name of employee,
last_name,string,Yes,,,Last name of employee,
age,integer,Yes,18,65,Employee age in years,
salary,decimal,Yes,50000,150000,Annual salary in USD,
hire_date,date,Yes,2020-01-01,2024-12-31,Date of hire (YYYY-MM-DD),
last_login_datetime,datetime,No,,,Last login timestamp (YYYY-MM-DD HH:MM:SS),
bonus_percentage,float,No,0,30,Annual bonus percentage,
department,string,Yes,,,Department name,"Engineering,Marketing,Sales,Finance,HR,Management"
is_active,boolean,Yes,,,Employment status (true/false),
skills,string,No,,,Semicolon-separated skills list,
email,string,Yes,,,Company email address,
phone,string,No,,,Contact phone number (+1-XXX-XXXX format),""",

    "CSV - Asian": """Column,Type,Required,Min,Max,Description,Allowed_Values
staff_id,integer,Yes,2000,2999,Staff identifier (2XXX series),
given_name,string,Yes,,,Given name,
family_name,string,Yes,,,Family name,
age,integer,Yes,22,60,Age in years,
monthly_salary,decimal,Yes,7000,12000,Monthly salary,
join_date,date,Yes,2019-01-01,2024-12-31,Date joined company,
last_activity,datetime,Yes,,,Last activity timestamp (ISO format),
performance_score,float,No,1,5,Performance rating (1-5 scale),
dept_code,string,Yes,,,Department code,"DEV,MKT,OPS,FIN,HR,MGT"
active_status,integer,Yes,0,1,Active status (1=active 0=inactive),
certifications,string,No,,,Semicolon-separated certifications,
work_email,string,Yes,,,Work email address,
mobile,string,No,,,Mobile phone with country code,""",

    "JSON - Mixed": """Field,DataType,Required,MinValue,MaxValue,Description,ValidValues
id,integer,true,3000,3999,Record identifier,
name.first,string,true,,,First name,
name.last,string,true,,,Last name,
age,integer,true,25,55,Age in years,
salary,number,true,65000,85000,Annual salary,
hired,date,true,2021-01-01,2024-12-31,Hire date,
active,boolean,true,,,Active status,
scores,array,false,,,Performance scores array,
department,string,true,,,Department,"Research,Engineering,Quality,Sales,Marketing"
""",

    "CSV - Clinical Trial": """Variable,DataType,Required,MinValue,MaxValue,Description,AllowedValues,Units
subject_id,string,Yes,,,Subject identifier (SXXX format),,
site_id,string,Yes,,,Clinical site ID,"SITE01,SITE02,SITE03,SITE04",
enrollment_date,date,Yes,2023-01-01,2024-12-31,Date of enrollment,,
visit_date,date,Yes,2023-01-01,2024-12-31,Date of visit,,
age,integer,Yes,18,85,Subject age,,years
gender,string,Yes,,,Biological gender,"M,F,X",
bmi,decimal,No,15,40,Body Mass Index,,kg/m²
treatment_arm,string,Yes,,,Treatment assignment,"Treatment,Placebo",
adverse_event,string,No,,,Adverse event description,,
lab_value,decimal,No,50,200,Primary lab result,,mg/dL
compliance_pct,decimal,Yes,0,100,Treatment compliance,,percent
completed_study,string,Yes,,,Study completion,"Y,N",
protocol_version,decimal,Yes,,,Protocol version,"2.0,2.1","""
}

def get_demo_dictionary(dataset_type: str) -> str:
    """Get the demo dictionary for a given dataset type"""
    return DEMO_DICTIONARIES.get(dataset_type, "")