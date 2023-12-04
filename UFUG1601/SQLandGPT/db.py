# Simple sqlite3 program
import sqlite3 as sq
sql = sq.connect('test.db')
# Create table for studets id and name and final grade
sql.execute('''CREATE TABLE STUDENTS 
         (ID INT PRIMARY KEY     NOT NULL,
         NAME           TEXT    NOT NULL,
         GRADE          INT     NOT NULL);''')
# Insert data into table
sql.execute("INSERT INTO STUDENTS (ID,NAME,GRADE) \
      VALUES (1, 'John', 90 )");    
sql.execute("INSERT INTO STUDENTS (ID,NAME,GRADE) \
        VALUES (2, 'Mary', 95 )");
sql.execute("INSERT INTO STUDENTS (ID,NAME,GRADE) \
        VALUES (3, 'Tom', 80 )");
# Put Sung 100 into table
sql.execute("INSERT INTO STUDENTS (ID,NAME,GRADE) \
        VALUES (4, 'Sung', 100 )");

# Commit changes
sql.commit()
# Select data from table from top 2 students
cursor = sql.execute("SELECT id, name, grade from STUDENTS ORDER BY grade")
for row in cursor:
   print ("ID = ", row[0])
   print ("NAME = ", row[1])
   
   
   print ("GRADE = ", row[2], "\n")
