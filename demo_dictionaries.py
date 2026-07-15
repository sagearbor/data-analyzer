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
protocol_version,decimal,Yes,,,Protocol version,"2.0,2.1",""",

    # Synthetic REDCap-style data dictionary (16 fields across 4 REDCap forms:
    # demographics, treatment, medical_history, safety), re-expressed in this
    # module's generic Column/Type/Required/Min/Max/Description/Allowed_Values
    # format so it loads through the same "load demo dictionary" quick-parse
    # path (web_app.py, ~line 1289) as every other entry in this dict, which
    # only understands that generic header - it does not build a 'fields' list
    # or invoke RuleExtractor/LogicValidator (see DataQualityAnalyzer in
    # web_app.py for where *real* REDCap uploads get branching-logic checks
    # via the LLM parser instead). Each field's original REDCap "Branching
    # Logic (Show field only if...)" condition is preserved in the
    # Description column so the relationship is still visible here even
    # though it isn't mechanically enforced by this quick-load path.
    # Source (16 fields, verbatim structure): mirrors
    # tests/test_data/dictionaries/synthetic/redcap_clinical_with_logic.csv
    # (not read at runtime - the Dockerfile does not COPY tests/, so the
    # content is embedded directly here to ship in the image).
    "REDCap - Clinical (synthetic, with branching logic)": """Column,Type,Required,Min,Max,Description,Allowed_Values
subject_id,string,Yes,,,Subject ID (synthetic test data) - fake identifier e.g. TEST-001,
age,integer,Yes,18,85,Age (years) - synthetic value,
gender,string,Yes,,,Biological Gender,"Male,Female,Other"
pregnant,string,No,,,"Currently Pregnant? Branching: shown only if gender='Female'","No,Yes"
weeks_pregnant,integer,No,0,42,"Weeks Pregnant. Branching: shown only if pregnant='Yes'",
due_date,date,No,,,"Expected Due Date (synthetic). Branching: shown only if pregnant='Yes'",
treatment_arm,string,Yes,,,Treatment Assignment,"Active Treatment,Placebo"
dosage_mg,decimal,No,5,500,"Dosage (mg), based on weight at 0.5mg/kg. Branching: shown only if treatment_arm='Active Treatment'",
placebo_type,string,No,,,"Placebo Formulation. Branching: shown only if treatment_arm='Placebo'","Tablet,Capsule"
diabetes,string,Yes,,,History of Diabetes?,"No,Yes"
diabetes_type,string,No,,,"Diabetes Type. Branching: shown only if diabetes='Yes'","Type 1,Type 2"
insulin_dependent,string,No,,,"Insulin Dependent? Branching: shown only if diabetes='Yes'","No,Yes"
lab_glucose,decimal,Yes,50,400,"Fasting Glucose (mg/dL), required if diabetic. Branching: shown only if diabetes='Yes'",
adverse_event,string,Yes,,,Any Adverse Events?,"No,Yes"
ae_description,string,No,,,"Describe Adverse Event (synthetic placeholder text). Branching: shown only if adverse_event='Yes'",
ae_severity,string,No,,,"Severity. Branching: shown only if adverse_event='Yes' and gender='Female'","Mild,Moderate,Severe,Life-threatening"
"""
}

def get_demo_dictionary(dataset_type: str) -> str:
    """Get the demo dictionary for a given dataset type"""
    return DEMO_DICTIONARIES.get(dataset_type, "")