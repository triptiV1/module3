#program for average rainfall using nested loop

month =0
avg_rain = 0
Total_rain = 0
number_of_years = int(input('Enter number of years  :')) #taking input for no. of years
for i in range(number_of_years):       #first FOR Loop
    print('This is the year #  ', i+1)
    for j in range(12):
        month = month +1   #second FOR loop
        #print(' This is month:', month)
        rainfall = float(input('Enter average rainfall in inches for this month'))
        Total_rain = rainfall + Total_rain
        print('Total rain till now = ', Total_rain)

    #Total_rain = rainfall+ Total_rain
    avg_rain = Total_rain / (number_of_years * 12)  #calculates average rainfall ove the entore time period entered

print('The total number of Months =  \n', number_of_years * 12)
print('The total inches of rainfall is ', Total_rain, end='')
print('inches')

print(f'The average rain per month over the period of {number_of_years *12} months is :  {avg_rain} inches')
