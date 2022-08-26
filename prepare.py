# .py file dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


'''function that does the following: 
- returns the endpoint class and topic
- takes in the dataframe and sets the date & time as
timestamp. function also creates new day & month columns
- labels ea. observation with
the Codeup program associated to the program_id
- handles missing values in the dataset ('endpoint' record)
- cleans specific "lesson/class" column values

# Codeup program_id to program type map:
# 1 = "full-stack PHP program"
# 2 = "full-stack JAVA program"
# 3 = "Data Science"
# 4 = "Front-end Program"
'''
def mass_log_clean(df):

    topics = df["endpoint"].str.split("/", n = 2, expand = True).rename(columns = {0: "class", 1: "topic"})
    topics = topics.drop(columns = 2)
    
    # combining the two(2) dataframes
    new_df = pd.concat([df, topics], axis = 1)

    # combining date and time & dropping previous columns
    new_df["datetime"] = new_df["date"] + " " + new_df["time"]

    # converting datetime column to proper pd.datetime 
    new_df["datetime"] = pd.to_datetime(new_df["datetime"])

    # setting the date column to index
    new_df = df.set_index("datetime").sort_index()

    new_df["program_type"] = new_df["program_id"].map(
        {1: "FS_PHP_program", 
        2: "FS_JAVA_program", 
        3: "DS_program", 
        4: "Front_End_program", 
        np.nan: None})

     # setting the program_id to object type
    new_df[["user_id", "program_id"]] = new_df[["user_id", "program_id"]].astype(object)

    # cleaning columns with empty class or nulls
    new_df = new_df.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', None)

    # dropping single record in 'endpoint' column 
    new_df = new_df.dropna(subset=['endpoint'])

    # Data Science program/class clean up
    new_df['class'] = np.where((new_df['class'] == 'fundamentals') & (df.program_type == 'DS_program'), 'ds_fundamentals', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '1-fundamentals') & (df.program_type == 'DS_program'), 'ds_fundamentals', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'storytelling') & (df.program_type == 'DS_program'), 'ds_storytelling', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '2-storytelling') & (df.program_type == 'DS_program'), 'ds_storytelling', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'sql') & (df.program_type == 'DS_program'), 'ds_sql', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '3-sql') & (df.program_type == 'DS_program'), 'ds_sql', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'python') & (df.program_type == 'DS_program'), 'ds_python', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '4-python') & (df.program_type == 'DS_program'), 'ds_python', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'stats') & (df.program_type == 'DS_program'), 'ds_stats', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '5-stats') & (df.program_type == 'DS_program'), 'ds_stats', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '6-regression') & (df.program_type == 'DS_program'), 'ds_regression', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'regression') & (df.program_type == 'DS_program'), 'ds_regression', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'classification') & (df.program_type == 'DS_program'), 'ds_classification', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '7-classification') & (df.program_type == 'DS_program'), 'ds_classification', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'clustering') & (df.program_type == 'DS_program'), 'ds_clustering', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '8-clustering') & (df.program_type == 'DS_program'), 'ds_clustering', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'timeseries') & (df.program_type == 'DS_program'), 'ds_timeseries', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '9-timeseries') & (df.program_type == 'DS_program'), 'ds_timeseries', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'anomaly-detection') & (df.program_type == 'DS_program'), 'ds_anomaly-detection', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '10-anomaly-detection') & (df.program_type == 'DS_program'), 'ds_anomaly-detection', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'nlp') & (df.program_type == 'DS_program'), 'ds_nlp', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '11-nlp') & (df.program_type == 'DS_program'), 'ds_nlp', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'distributed-ml') & (df.program_type == 'DS_program'), 'ds_distributed-ml', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '12-distributed-ml') & (df.program_type == 'DS_program'), 'ds_distributed-ml', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'advanced-topics') & (df.program_type == 'DS_program'), 'ds_advanced-topics', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '13-advanced-topics') & (df.program_type == 'DS_program'), 'ds_advanced-topics', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == 'appendix') & (df.program_type == 'DS_program'), 'ds_appendix', new_df['class'])
    
    # the remaining class names will be Full-Stack program
    new_df['class'] = np.where((new_df['class'] == '1-fundamentals'), 'fundamentals', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '10-anomaly-detection'), 'anomaly-detection', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '6-regression'), 'regression', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '3-sql'), 'sql', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '4-python'), 'python', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '5-stats'), 'stats', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '7-classification'), 'classification', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '2-storytelling'), 'storytelling', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '9-timeseries'), 'timeseries', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '8-clustering'), 'clustering', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '11-nlp'), 'nlp', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '12-distributed-ml'), 'distributed-ml', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '1._Fundamentals'), 'fundamentals', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '5-regression'), 'regression', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '6-classification'), 'classification', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '9-anomaly-detection'), 'anomaly-detection', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '3.0-mysql-overview'), 'mysql', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '13-advanced-topics'), 'advanced-topics', new_df['class'])
    new_df['class'] = np.where((new_df['class'] == '2-stats'), 'stats', new_df['class'])

    # returns the new df w/endpoint class and topics
    return new_df



'''function to loop through discrete variables and prints variable, 
data type, number of unique observations, unique observations, and 
frequency of unique observations sorted (value_counts)'''
def print_variable_info(df):
    with pd.option_context("display.max_rows", None):
        for col in df.columns:
            if df[col].dtype != "number":
                print(f'feature: {col}')
                print(f'feature type: {df[col].dtype}')
                print(f'number of unique values: {df[col].nunique()}')
                print(f'unique values: {df[col].unique()}')
                print(f'value counts: {df[col].value_counts().sort_index(ascending = True).sort_values(ascending = False)}')


'''functiont to create a dataframe that captures the highest frequency value per column'''
def print_frequency_df(df):
    # container to hold metrics from for loop
    container = []

    for col in df.columns:

        metric = {  
            "feature": col,
            "data_type": df[col].dtype,
            "unique_values": df[col].nunique(),
            "most_freq_observation": df[col].value_counts().idxmax(),
            "total_observations": df[col].value_counts().max()
        }

        container.append(metric)

    freq_df = pd.DataFrame(container).sort_values("total_observations", ascending = False)

    return freq_df


'''function to deal with parsing one entry in our log data'''
def parse_log_entry(entry):
    parts = entry.split()
    output = {}
    output['ip'] = parts[0]
    output['timestamp'] = parts[3][1:].replace(':', ' ', 1)
    output['request_method'] = parts[5][1:]
    output['request_path'] = parts[6]
    output['http_version'] = parts[7][:-1]
    output['status_code'] = parts[8]
    output['size'] = int(parts[9])
    output['user_agent'] = ' '.join(parts[11:]).replace('"', '')
    return pd.Series(output)


'''function that returns a df of unique null feature records'''
def get_fifty_3(df):

    new_df = df.loc[df[[
        "name", 
        "slack", 
        "start_date", 
        "end_date", 
        "program_type"]].isnull().apply(lambda x: all(x), axis=1)]

    return new_df

'''Function that plots the post-grad topic revisits'''
def most_grad_revisits(df):

    grad = df.copy()

    # list of programs to plot (exludes null values)
    lst = ['FS_PHP_program', 'FS_JAVA_program', 'Front_End_program', 'DS_program']
    
    for program in lst:

        if program != "Front_End_program":
            plt.figure(figsize = (8, 3))
            sns.set(font_scale = 0.7)

            df1 = grad[grad["program_type"] == program]

            df1 = df1.applymap(lambda s: s.capitalize() if type(s) == str else s)

            sns.countplot(
                y = "class", 
                data = df1,
                order = df1["class"].value_counts()[0:11].index,
                palette = "crest_r")

            plt.ylabel(None)
            plt.xlabel(None)
            plt.title(f'Top 10 Revisited Lessons Post Graduation: {program}')
            plt.show()

'''function to plot most revisited topics for alumni'''
def most_grad_revisits_topics(df):

    grad = df.copy()

    # list of programs to plot (exludes null values)
    lst = ['FS_PHP_program', 'FS_JAVA_program', 'Front_End_program', 'DS_program']
    
    for program in lst:

        if program != "Front_End_program":
            plt.figure(figsize = (8, 3))
            sns.set(font_scale = 0.7)

            df1 = grad[grad["program_type"] == program]

            df1 = df1.applymap(lambda s: s.capitalize() if type(s) == str else s)

            sns.countplot(
                y = "topic", 
                data = df1,
                order = df1["topic"].value_counts()[0:11].index,
                palette = "crest_r")

            plt.ylabel(None)
            plt.xlabel(None)
            plt.title(f'Top 10 Revisited Topics Post Graduation: {program}')
            plt.show()

'''Function that plots the current topic revisits'''
def most_current_visits(df):

    curr = df.copy()

    # list of programs to plot (exludes null values)
    lst = ['FS_JAVA_program', 'DS_program']
    
    for program in lst:
        
        plt.figure(figsize = (8, 3))
        sns.set(font_scale = 0.7)

        df1 = curr[curr["program_type"] == program]

        df1 = df1.applymap(lambda s: s.capitalize() if type(s) == str else s)

        sns.countplot(
            y = "class", 
            data = df1,
            order = df1["class"].value_counts()[0:11].index,
            palette = "crest_r")

        plt.ylabel(None)
        plt.xlabel(None)
        plt.title(f'While Enrolled: Top-10 Lessons Visited: {program}')
        plt.show()


def value_counts_and_frequencies(s: pd.Series, dropna=True) -> pd.DataFrame:
    return pd.merge(
        s.value_counts(dropna=True)[0:6].rename('Count'),
        s.value_counts(dropna=True, normalize=True)[0:6].rename('Percentage').round(2),
        left_index=True,
        right_index=True,
    )


'''function that returns the top 30 most frequent classes as a plot'''
def return_most_visited_lessons_all_time(df):
    
    plt.figure(figsize=(12, 8))
    sns.set(font_scale = 0.7)

    sns.countplot(
        y = "class", 
        data = df,
        order = df["class"].value_counts(dropna = True)[0:31].index,
        palette = "crest_r")

    plt.ylabel(None)
    plt.xlabel("Number of Visits")
    plt.title("Most Explored Codeup Lessons: All Time")
    plt.show()



def prep(df, user):
    df = df[df.user_id == user]
    pages = df['endpoint'].resample('d').count()
    return pages


def compute_pct_b(pages, span, weight, user):
    midband = pages.ewm(span=span).mean()
    stdev = pages.ewm(span=span).std()
    ub = midband + stdev*weight
    lb = midband - stdev*weight
    bb = pd.concat([ub, lb], axis=1)
    my_df = pd.concat([pages, midband, bb], axis=1)
    my_df.columns = ['pages', 'midband', 'ub', 'lb']
    my_df['pct_b'] = (my_df['pages'] - my_df['lb'])/(my_df['ub'] - my_df['lb'])
    my_df['user_id'] = user
    return my_df

def plt_bands(my_df, user):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(my_df.index, my_df.pages, label='Number of Pages, User: '+str(user))
    ax.plot(my_df.index, my_df.midband, label = 'EMA/midband')
    ax.plot(my_df.index, my_df.ub, label = 'Upper Band')
    ax.plot(my_df.index, my_df.lb, label = 'Lower Band')
    ax.legend(loc='best')
    ax.set_ylabel('Number of Pages')
    plt.show()

def find_anomalies(df, user, span, weight):
    pages = prep(df, user)
    my_df = compute_pct_b(pages, span, weight, user)
    # plt_bands(my_df, user)
    return my_df[my_df.pct_b>1]