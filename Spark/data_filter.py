from pyspark import SparkContext, SparkConf
import math
from operator import add
import csv
from io import StringIO

config = SparkConf().setMaster("local").setAppName("my_app")
sp_context = SparkContext(conf=config)

#---------------------------------------------------AUTO DATA-----------------------------------------------------------


def load_record_1(line):
    data_input = StringIO(line)
    reader = csv.DictReader(data_input, fieldnames=["X", "Y", "OBJECTID", "ID", "NAME_1"
        , "NAME_2", "ADDRESS", "CITY", "ZIPCODE", "STATE", "EMPLOYMENT", "YEARBUILT", "SQUAREFEET"
        , "SOURCE", "DATEREVISE", "COMMENTS"])
    return next(reader)


def write_record_1(records):
    data_output = StringIO()
    writer = csv.DictWriter(data_output, fieldnames=["X", "Y", "OBJECTID", "ID", "NAME_1"
        , "NAME_2", "ADDRESS", "CITY", "ZIPCODE", "STATE", "EMPLOYMENT", "YEARBUILT", "SQUAREFEET"
        , "SOURCE", "DATEREVISE", "COMMENTS"])
    for record in records:
        writer.writerow(record)
    return [data_output.getvalue()]

#---------------Process Data-----------------

data_1 = sp_context.textFile("Automotive_Facilities.csv").map(load_record_1)

ford_rdd = data_1.filter(lambda x: x["NAME_1"] == "Ford")
gm_rdd = data_1.filter(lambda x: x["NAME_1"] == "General Motors")
ch_rdd = data_1.filter(lambda x: x["NAME_1"] == "Chrysler")

emp_rdd_1 = data_1.map(lambda x: x["EMPLOYMENT"])
emp_rdd_2 = emp_rdd_1.filter(lambda x: x != "EMPLOYMENT")
emp_rdd = emp_rdd_2.map(lambda x: int(x))

emp_rdd_group_1 = emp_rdd.filter(lambda x: (x >= 0) & (x <= 2000))
emp_rdd_group_2 = emp_rdd.filter(lambda x: (x >= 2001) & (x <= 4000))
emp_rdd_group_3 = emp_rdd.filter(lambda x: (x >= 4001) & (x <= 6000))
emp_rdd_group_4 = emp_rdd.filter(lambda x: (x >= 6001) & (x <= math.inf))


#List Number of Facilities Per Company
# Ford
print("Ford: ", ford_rdd.count())
# Chrysler
print("Chrysler: ", ch_rdd.count())
# GM
print("GM", gm_rdd.count(), "\n")


# [0, 2000] employees
print("[0, 2000]: ", emp_rdd_group_1.count())
# [2001, 4000] employees
print("[0, 2000]: ", emp_rdd_group_2.count())
# [4001, 6000] employees
print("[0, 2000]: ", emp_rdd_group_3.count())
# (6000, inf) employees
print("[0, 2000]: ", emp_rdd_group_1.count(), "\n")

#------------------------------------------------------SALES DATA-------------------------------------------------------


def load_record_2(line):
    data_input = StringIO(line)
    reader = csv.DictReader(data_input, fieldnames=["Region", "Country", "Item Type", "Sales Channel", "Order Priority"
        , "Order Date", "Order ID", "Ship Date", "Units Sold", "Unit Price", "Unit Cost", "Total Revenue", "Total Cost"
        , "Total Profit"])
    return next(reader)


def write_record_2(records):
    data_output = StringIO()
    writer = csv.DictWriter(data_output, fieldnames=["Region", "Country", "Item Type", "Sales Channel", "Order Priority"
        , "Order Date", "Order ID", "Ship Date", "Units Sold", "Unit Price", "Unit Cost", "Total Revenue", "Total Cost"
        , "Total Profit"])
    for record in records:
        writer.writerow(record)
    return [data_output.getvalue()]


#---------------Process Data-----------------

data_2 = sp_context.textFile("100000_Sales_Records.csv").map(load_record_2)

sales_rdd_1 = data_2.map(lambda x: [x["Sales Channel"], x["Total Profit"]])

sales_rdd_1_online_1 = sales_rdd_1.filter(lambda x: x[0] == "Online")
sales_rdd_1_offline_1 = sales_rdd_1.filter(lambda x: x[0] == "Offline")

sales_rdd_1_online_2 = sales_rdd_1_online_1.map(lambda x: x[1])
sales_rdd_1_online = sales_rdd_1_online_2.map(lambda x: float(x))
sales_rdd_1_offline_2 = sales_rdd_1_offline_1.map(lambda x: x[1])
sales_rdd_1_offline = sales_rdd_1_offline_2.map(lambda x: float(x))


sales_rdd_2 = data_2.map(lambda x: [x["Sales Channel"], x["Order Priority"], x["Total Profit"]])

sales_rdd_2_online = sales_rdd_2.filter(lambda x: x[0] == "Online")
sales_rdd_2_online_H_1 = sales_rdd_2_online.filter(lambda x: x[1] == "H")
sales_rdd_2_online_H_2 = sales_rdd_2_online_H_1.map(lambda x: x[2])
sales_rdd_2_online_H = sales_rdd_2_online_H_2.map(lambda x: float(x))
sales_rdd_2_online_M_1 = sales_rdd_2_online.filter(lambda x: x[1] == "M")
sales_rdd_2_online_M_2 = sales_rdd_2_online_M_1.map(lambda x: x[2])
sales_rdd_2_online_M = sales_rdd_2_online_M_2.map(lambda x: float(x))
sales_rdd_2_online_L_1 = sales_rdd_2_online.filter(lambda x: x[1] == "L")
sales_rdd_2_online_L_2 = sales_rdd_2_online_L_1.map(lambda x: x[2])
sales_rdd_2_online_L = sales_rdd_2_online_L_2.map(lambda x: float(x))
sales_rdd_2_online_C_1 = sales_rdd_2_online.filter(lambda x: x[1] == "C")
sales_rdd_2_online_C_2 = sales_rdd_2_online_C_1.map(lambda x: x[2])
sales_rdd_2_online_C = sales_rdd_2_online_C_2.map(lambda x: float(x))

sales_rdd_2_offline = sales_rdd_2.filter(lambda x: x[0] == "Offline")
sales_rdd_2_offline_H_1 = sales_rdd_2_offline.filter(lambda x: x[1] == "H")
sales_rdd_2_offline_H_2 = sales_rdd_2_offline_H_1.map(lambda x: x[2])
sales_rdd_2_offline_H = sales_rdd_2_offline_H_2.map(lambda x: float(x))
sales_rdd_2_offline_M_1 = sales_rdd_2_offline.filter(lambda x: x[1] == "M")
sales_rdd_2_offline_M_2 = sales_rdd_2_offline_M_1.map(lambda x: x[2])
sales_rdd_2_offline_M = sales_rdd_2_offline_M_2.map(lambda x: float(x))
sales_rdd_2_offline_L_1 = sales_rdd_2_offline.filter(lambda x: x[1] == "L")
sales_rdd_2_offline_L_2 = sales_rdd_2_offline_L_1.map(lambda x: x[2])
sales_rdd_2_offline_L = sales_rdd_2_offline_L_2.map(lambda x: float(x))
sales_rdd_2_offline_C_1 = sales_rdd_2_offline.filter(lambda x: x[1] == "C")
sales_rdd_2_offline_C_2 = sales_rdd_2_offline_C_1.map(lambda x: x[2])
sales_rdd_2_offline_C = sales_rdd_2_offline_C_2.map(lambda x: float(x))


country_prof_rdd_1 = data_2.map(lambda x: [x["Country"], x["Total Profit"]]).sortByKey()
country_prof_rdd_2 = country_prof_rdd_1.filter(lambda x: x[0] != "Country").map(lambda x: [x[0], float(x[1])])
country_prof_rdd_3 = country_prof_rdd_2.reduceByKey(add)
country_prof_rdd = country_prof_rdd_3.sortBy(lambda x: x[1], ascending=False)

country_rev_rdd_1 = data_2.map(lambda x: [x["Country"], x["Total Revenue"]]).sortByKey()
country_rev_rdd_2 = country_rev_rdd_1.filter(lambda x: x[0] != "Country").map(lambda x: [x[0], float(x[1])])
country_rev_rdd_3 = country_rev_rdd_2.reduceByKey(add)
country_rev_rdd = country_rev_rdd_3.sortBy(lambda x: x[1], ascending=False)


# top 5 totoal rev.
print("Top 5 Revenue(by Country): ")
print(country_rev_rdd.take(5))
# top 5 profits
print("Top 5 Profit(by Country): ")
print(country_prof_rdd.take(5), "\n")
# total profit (online sales)
print("Total Profit(Online): ", sales_rdd_1_online.sum())
# total profit (offline sales)
print("Total Profit(Offline): ", sales_rdd_1_offline.sum(), "\n")
# total profit (online sales)
# H
print("Total Profit(Online, H):", sales_rdd_2_online_H.sum())
# M
print("Total Profit(Online, M):", sales_rdd_2_online_M.sum())
# L
print("Total Profit(Online, L):", sales_rdd_2_online_L.sum())
# C
print("Total Profit(Online, C):", sales_rdd_2_online_C.sum())

# total profit (offline sales)
# H
print("Total Profit(Offline, H):", sales_rdd_2_offline_H.sum())
# M
print("Total Profit(Offline, M):", sales_rdd_2_offline_M.sum())
# L
print("Total Profit(Offline, L):", sales_rdd_2_offline_L.sum())
# C
print("Total Profit(Offline, C):", sales_rdd_2_offline_C.sum())