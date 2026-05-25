from tkinter import *
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tkinter import filedialog, END
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

global X_train, X_test, y_train, y_test 
def upload_dataset():
    global df
    # Open file dialog to select .csv file
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        # Read the CSV file using pandas
        df = pd.read_csv(file_path)
        
        # Get the shape of the dataset
        shape_info = f"Shape of dataset: {df.shape}\n\n"
        
        # Get the top 5 rows
        top_5 = df.head()
        
        # Clear the text box and insert the information
        text.delete(1.0, END)
        text.insert(INSERT, "Top 5 Rows:\n")
        text.insert(INSERT, top_5.to_string(index=False))
        text.insert(INSERT, shape_info)

def preprocess_data():
    global df
    # Preprocessing steps
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    df[np.isnan(df['TotalCharges'])]
    df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
    df.fillna(df["TotalCharges"].mean(), inplace=True)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    
    # You can print or display these in the text box
    description = df["InternetService"].describe(include=['object', 'bool'])
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    text.delete(1.0, END)
    
    # Display information in the text box
    text.delete(1.0, END)
    text.insert(INSERT, f"Preprocessed Data:\n{description}\n\n")
    text.insert(INSERT, f"Numerical Columns:\n{df[numerical_cols].describe()}\n\n")
    text.insert(INSERT, f"Preprocessed Data:\n{df.head()}")

def graph_1():
    global df
    g_labels = ['Male', 'Female']
    c_labels = ['No', 'Yes']
    
    # Gender Pie Chart
    gender_counts = df['gender'].value_counts()
    churn_counts = df['Churn'].value_counts()
    
    # Define color palettes
    gender_colors = ['#1f77b4', '#ff7f0e']  # Blue for Male, Orange for Female
    churn_colors = ['#2ca02c', '#d62728']  # Green for No Churn, Red for Yes Churn
    
    # Create a figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=120)

    # Plot gender distribution
    wedges, texts, autotexts = axes[0].pie(
        gender_counts, labels=g_labels, autopct='%1.1f%%', startangle=90, colors=gender_colors,
        wedgeprops=dict(width=0.3, edgecolor="w"), textprops={'size': 14, 'weight': 'bold'})
    axes[0].set_title("Gender Distribution", fontsize=16, weight='bold')

    # Plot churn distribution
    wedges, texts, autotexts = axes[1].pie(
        churn_counts, labels=c_labels, autopct='%1.1f%%', startangle=90, colors=churn_colors,
        wedgeprops=dict(width=0.3, edgecolor="w"), textprops={'size': 14, 'weight': 'bold'})
    axes[1].set_title("Churn Distribution", fontsize=16, weight='bold')

    # Add a centered title for the entire figure
    fig.suptitle('Customer Gender and Churn Distribution', fontsize=18, weight='bold')

    # Show the legend
    axes[1].legend(wedges, c_labels, title="Churn", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Clear the text box
    text.delete(1.0, END)
    
    # Additional calculations for churn by gender
    churn_by_gender_no = df["Churn"][df["Churn"]=="No"].groupby(by=df["gender"]).count()
    churn_by_gender_yes = df["Churn"][df["Churn"]=="Yes"].groupby(by=df["gender"]).count()
    
    # Insert the churn information into the text box
    text.insert(INSERT, "Churn by gender (No):\n")
    text.insert(INSERT, churn_by_gender_no.to_string() + "\n\n")
    text.insert(INSERT, "Churn by gender (Yes):\n")
    text.insert(INSERT, churn_by_gender_yes.to_string() + "\n\n")
def graph_2():
    # Clear the text box for any previous output
    text.delete(1.0, END)

    # Insert some textual information about the graph (optional)
    text.insert(INSERT, "Churn Distribution w.r.t Gender: Male (M), Female (F)\n")
    text.insert(INSERT, "This chart displays the gender-wise churn distribution.\n\n")

    # Data for the pie charts
    labels = ["Churn: Yes", "Churn: No"]
    values = [1869, 5163]
    labels_gender = ["F", "M", "F", "M"]
    sizes_gender = [939, 930, 2544, 2619]
    colors = ['#ff6666', '#66b3ff']
    colors_gender = ['#c2c2f0', '#ffb3e6', '#c2c2f0', '#ffb3e6']
    explode = (0.3, 0.3)  # Explode for Churn pie chart
    explode_gender = (0.1, 0.1, 0.1, 0.1)  # Explode for Gender pie chart
    textprops = {"fontsize": 15}  # Set font size for text in the pie charts

    # Plotting the pie charts
    plt.figure(figsize=(6, 6))

    # Churn distribution pie chart
    plt.pie(values, labels=labels, autopct='%1.1f%%', pctdistance=1.08, labeldistance=0.8,
            colors=colors, startangle=90, frame=True, explode=explode, radius=10,
            textprops=textprops, counterclock=True)

    # Gender distribution pie chart
    plt.pie(sizes_gender, labels=labels_gender, colors=colors_gender, startangle=90,
            explode=explode_gender, radius=7, textprops=textprops, counterclock=True)

    # Draw circle in the middle to make it look like a donut chart
    centre_circle = plt.Circle((0, 0), 5, color='black', fc='white', linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Title for the pie charts
    plt.title('Churn Distribution w.r.t Gender: Male(M), Female(F)', fontsize=15, y=1.1)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def graph_3():
    global df
    # Clear the text box for any previous output
    text.delete(1.0, END)

    # Insert some textual information about the graph (optional)
    text.insert(INSERT, "Customer Contract Distribution\n")
    text.insert(INSERT, "This bar chart shows the distribution of customer contracts categorized by Churn.\n\n")
    # Create a grouped bar chart for 'Churn' vs 'Contract'
    contract_churn = pd.crosstab(df['Churn'], df['Contract'])

    # Plotting the grouped bar chart
    contract_churn.plot(kind='bar', figsize=(10, 6), width=0.8)

    # Adding title and labels
    plt.title('Customer Contract Distribution with respect to Churn', fontsize=15)
    plt.xlabel('Churn', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)

    # Display the plot
    plt.tight_layout()
    plt.show()

def graph_4():
    global df, text  # Add 'text' here to reference the global widget
    # Clear the text box for any previous output
    text.delete(1.0, END)

    # Insert some textual information about the graph (optional)
    text.insert(INSERT, "Payment Method Distribution\n")
    text.insert(INSERT, "This pie chart shows the distribution of different payment methods used by customers.\n\n")

    # Data for the pie chart
    labels = df['PaymentMethod'].unique()
    values = df['PaymentMethod'].value_counts()

    # Define color palette for more vibrant colors
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']

    # Create the pie chart using matplotlib
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1.5, 'linestyle': 'solid'},
                                       textprops={'fontsize': 14, 'weight': 'bold'})

    # Enhance the look with more customizations
    for text in autotexts:
        text.set_color('white')  # Make percentage text white for contrast

    # Title with enhanced styling
    plt.title("Payment Method Distribution", fontsize=18, weight='bold', color='darkblue')

    # Add a shadow effect to the pie chart
    plt.gca().set_facecolor('lightgray')

    # Display the plot with a circular aspect ratio to keep it balanced
    plt.axis('equal')

    # Show the plot
    plt.show()
   


def object_to_int(dataframe_series):
    if dataframe_series.dtype == 'object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

def train_test_split_function():
    global df  # Ensure you're working with the global dataframe

    # Convert all object columns to integers using LabelEncoder
    df = df.apply(lambda x: object_to_int(x))

    # Separate features (X) and target (y)
    X = df.drop(columns=['Churn'])
    y = df['Churn'].values

    # Split the data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, stratify=y)

    text.delete(1.0, END)
    text.insert(INSERT, "Training Data Shapes:\n")
    text.insert(INSERT, f"X_train shape: {X_train.shape}\n")
    text.insert(INSERT, f"y_train shape: {y_train.shape}\n")
    text.insert(INSERT, f"X_test shape: {X_test.shape}\n")
    text.insert(INSERT, f"y_test shape: {y_test.shape}\n")
    text.insert(INSERT, f"y_test shape: {X_train}\n")






# This function will train and evaluate the models and store their accuracy scores
def algorithms():
    global df, text,X_train, X_test, y_train, y_test   # Ensure you're working with the global dataframe

    # Prepare features (X) and target (y)
    X = df.drop(columns=['Churn'])
    y = df['Churn'].values

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, stratify=y)

    # Initialize the models
    models = {
        "XGBoost": xgb.XGBClassifier(),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "SVM": SVC(),
        "Decision Trees": DecisionTreeClassifier(),
    }

    # Initialize a dictionary to hold the results
    results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        # Store the results in the dictionary
        results[model_name] = {
            "Accuracy": accuracy*100,
            "Precision": precision*100,
            "Recall": recall*100,
            "F1-Score": f1*100
        }

    # Clear the text box before displaying new results
    text.delete(1.0, END)

    # Insert the results into the text box
    text.insert(INSERT, "Model Evaluation Results:\n\n")
    for model_name, metrics in results.items():
        text.insert(INSERT, f"{model_name}:\n")
        text.insert(INSERT, f"Accuracy: {metrics['Accuracy']:.4f}\n")
        text.insert(INSERT, f"Precision: {metrics['Precision']:.4f}\n")
        text.insert(INSERT, f"Recall: {metrics['Recall']:.4f}\n")
        text.insert(INSERT, f"F1-Score: {metrics['F1-Score']:.4f}\n\n")

    # Store the accuracy results globally for later plotting
    global accuracy_scores
    accuracy_scores = {model_name: metrics['Accuracy'] for model_name, metrics in results.items()}




# This function will plot the accuracy scores of the models in a more attractive bar chart
def show_accuracy_graph():
    global accuracy_scores  # Use the global variable storing the accuracy scores

    # If no accuracy scores are available, return early
    if not accuracy_scores:
        text.delete(1.0, END)
        text.insert(INSERT, "No accuracy scores available. Please run the algorithms first.")
        return

    # Create a bar chart for the accuracy scores
    models = list(accuracy_scores.keys())
    accuracies = list(accuracy_scores.values())

    # Define the position of each bar
    x_pos = np.arange(len(models))

    # Set up the figure and axis
    plt.figure(figsize=(10, 6))

    # Use a gradient color map for bars
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(np.linspace(0.2, 0.8, len(models)))

    # Create bars with customized color, width, and edge color
    bars = plt.bar(x_pos, accuracies, color=colors, edgecolor='black', width=0.6)

    # Add labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Title and labels with custom font size and color
    plt.title('Accuracy Scores of Different Algorithms', fontsize=16, fontweight='bold', color='darkblue')
    plt.xlabel('Models', fontsize=14, fontweight='bold', color='darkgreen')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold', color='darkgreen')

    # Adding horizontal gridlines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Customize the x-axis ticks with the model names and adjust the font size
    plt.xticks(x_pos, models, rotation=45, ha="right", fontsize=12)

    # Set the limits for the y-axis to be from 0 to 1
    plt.ylim(0, 100)

    # Show the plot with tight layout
    plt.tight_layout()
    plt.show()


def predict():
    global rf, X_train, y_train  # Use the trained RandomForestClassifier model
    global df  # Access the global dataframe for feature reference
    
    # Train RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        text.delete(1.0, END)
        text.insert(INSERT, "No file selected.")
        return
    
    try:
        # Load the data for prediction
        input_data = pd.read_csv(file_path)
        
        # Ensure the input data has the same columns as the training data
        feature_columns = df.drop(columns=['Churn']).columns
        input_data = input_data[feature_columns]
        
        # Preprocess the input data
        input_data = input_data.apply(lambda x: object_to_int(x))
        
        # Make predictions
        predictions = rf.predict(input_data)
        
        # Convert predictions to "Yes" or "No"
        converted_predictions = ["Yes" if pred == 1 else "No" for pred in predictions]
        
        # Display predictions in the text box
        text.delete(1.0, END)
        text.insert(INSERT, "Predictions:\n")
        for idx, pred in enumerate(converted_predictions, start=1):
            text.insert(INSERT, f"Row {idx}: {pred}\n")
    
    except Exception as e:
        text.delete(1.0, END)
        text.insert(INSERT, f"Error occurred during prediction: {e}")



main = tk.Tk()
main.title("Customer Churn Prediction using Machine Learning Models") 
main.geometry("1600x1500")

# Configure the main grid
main.grid_columnconfigure(0, weight=1)
main.grid_rowconfigure(1, weight=1)

# Title
font_title = ('Arial', 20, 'bold underline')

# Enhanced title with padding and border
title = Label(
    main, 
    text='Customer Churn Prediction using Machine Learning Models', 
    font=font_title,
    bg='#00274D',  # A darker blue for background
    fg='#FFD700',  # Gold color for text
    relief="ridge",  # Add a 3D effect
    bd=10,  # Border width
    padx=20,  # Add internal horizontal padding
    pady=10  # Add internal vertical padding
)

# Adjust title layout
title.grid(row=0, column=0, sticky="ew", padx=15, pady=15)

# Navigation Bar
nav_frame = Frame(main, bg="gray")
nav_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

font_nav = ('times', 13, 'bold')
nav_buttons = [
    {"text": "Upload Dataset", "bg": "sky blue", "command": upload_dataset},
    {"text": "Preprocessing", "bg": "light green", "command": preprocess_data},
    {"text": "Graph_1", "bg": "turquoise", "command": graph_1},
    {"text": "Graph_2", "bg": "lightblue","command":graph_2},
    {"text": "Graph_3", "bg": "lightyellow","command":graph_3},
    {"text": "Train_test_Split", "bg": "coral","command":train_test_split_function},
    {"text": "Algorithms", "bg": "gold", "command": algorithms},
    {"text": "Show Accuracy Graph", "bg": "violet","command":show_accuracy_graph},
    {"text": "Predict", "bg": "green","command":predict},
]

# Add buttons to the first row (index 0) and second row (index 1)
for i, button in enumerate(nav_buttons):
    nav_frame.grid_columnconfigure(i, weight=1)
    row = i // 5  # Determines which row to place the button in
    column = i % 5  # Determines the column in the row
    btn = Button(nav_frame, text=button["text"], bg=button["bg"], font=font_nav, command=button.get("command"))
    btn.grid(row=row, column=column, padx=5, pady=5, sticky="ew")

# Text Box for Logs
font1 = ('times', 12, 'bold')
text_frame = Frame(main)
text_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
main.grid_rowconfigure(2, weight=1)

text = Text(text_frame, wrap="word", font=font1)
scroll = Scrollbar(text_frame, orient="vertical", command=text.yview)
text.configure(yscrollcommand=scroll.set)

scroll.pack(side="right", fill="y")
text.pack(side="left", fill="both", expand=True)

main.config(bg='skyblue')
main.mainloop()