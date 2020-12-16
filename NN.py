import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template
from sklearn import linear_model


sdfsddasdasdasasdasdasdasda

font1 = {
    'family': 'Times New Roman',
    # 'weight': 'bold',
    'size': 20
    # 'style': 'italic'
    }

np.set_printoptions(suppress=True)

def sigmoid(x):
    return 1/(1+np.exp(-x))

'''
A three layer neuron network (input-hidden-output)
the hidden layer contains four neurons which requires 25 input values for each (24 variables plus 1 bias)
the activation function of the first layer is the sigmoid function
'''
coeff_hidden = np.array(
                    [
                        0.585589581601409, 0.546041033198155, 0.0744665550442488, 0.00339273378695232,
                        -8.41987941873862, 0.0498252012944573, 0.419436673610623, 0.218636610656515,
                        0.391366070758325, 0.262006650863285, 0.0271145423644438, -0.0220495869392937,
                        0.435867182088122, 0.227811362593927, 0.36645699365062, -0.052982313758509,
                        0.254789325998335, 1.40910295857932, 1.16721434656853, -0.114837799757919,
                        0.370282055580022, -1.57503771471018, -1.88633640437711, -1.29966678547714,
                        -0.899649963000579, 1.76525741654151, 0.725044154294771, 0.132252166411055,
                        0.273395116588222, -2.22016083760696, 0.173485957191131, 0.225675589308854,
                        0.525304896071524, 2.09916767542293, 2.20011495475831,  2.08611590454957,
                        1.11736896635433, 1.22051954050451, 0.844575113110987,  0.31487184440679,
                        -0.586558276937015, 0.160236303607634, -0.0610459573286011, -1.29577985546957,
                        0.0638189508142904, 1.09199514387195, -0.790711960119485, 0.645596043000561,
                        1.58893681064465, -0.0801013032888691, 1.3065064270667, 0.717898290044641,
                        0.0538550323013031, -0.636810896223117, -2.24250770848636, 0.159736336975371,
                        0.387686008850728, 0.46556700524455, 0.849191853548704, 0.283938409787756,
                        0.159708155120536, -0.618902068388406, 0.203374899017394, 0.478130506012486,
                        0.273899654770796, -0.286712012565712, 0.932608595747422, 1.4793424383781,
                        0.707600537396147, -0.335262888558835, 1.0152769466719, -0.662975663423879,
                        -1.01956065225956, -0.2549512188683, -0.908855020434902, 0.750482423865247,
                        -0.679801179493518, -0.0468020024041587, 0.532717716703409, 0.563803365505634,
                        -0.121449677892823, -0.441130411096466, -0.445497408940753, -0.427446585977514,
                        0.148668144589363, 0.282199704299667, 0.616935053076751, -0.316719597882571,
                        -0.327385875443779, -0.326680397776587, -0.00882311260724723, -0.930272453213385,
                        -2.31239819039686, -1.87863264107556, 0.0697321381438693, -0.831980775312759,
                        1.17570636483139, 0.40447637977899, 0.259566097643218, 0.663083389515968
                    ]).reshape(4, 25).transpose()
coeff_output = np.array(
                    [
                        0.344367782982545, -0.405330194842118, 1.45792396970537, -1.09562620311941, -0.901616165011739
                    ])

#### implement the meta-model

input_data = pd.read_csv(r'D:\My Drive\Digital_Agriculture\PepsiCo\yield model\S2.csv')
# input_data2 = pd.read_csv(r'D:\My Drive\Digital_Agriculture\PepsiCo\sowing_date\UK\sensitivity_analysis\-5.csv')
input_data2 = pd.read_csv(r'D:\My Drive\Digital_Agriculture\PepsiCo\yield model\S2.csv')
## process input data
input = Template(
    "["
    "{"
                 "'SowDay':{{SowDay}}, "
                 "'SoilWC': {{SoilWC}},"
                 "'NRate': {{NRate}},"
                 "'Tillers': {{Tillers}},"
                 "'Rain00': {{Rain00}},"
                 "'Rain01': {{Rain01}},"
                 "'Rain02': {{Rain02}},"
                 "'Rain03': {{Rain03}},"
                 "'Rain04': {{Rain04}},"
                 "'Rain05': {{Rain05}},"
                 "'Rain06': {{Rain06}},"
                 "'Rain07': {{Rain07}},"
                 "'Rain08': {{Rain08}},"
                 "'Rain09': {{Rain09}},"
                 "'Rad00': {{Rad00}},"
                 "'Rad01': {{Rad01}},"
                 "'Rad02': {{Rad02}},"
                 "'Rad03': {{Rad03}},"
                 "'Rad04': {{Rad04}},"
                 "'Rad05': {{Rad05}},"
                 "'Rad06': {{Rad06}},"
                 "'Rad07': {{Rad07}},"
                 "'Rad08': {{Rad08}},"
                 "'Rad09': {{Rad09}},"
"}"
"]")
score_train = np.zeros(201)
for index, nrate in enumerate(range(-100, 101)):
    fitYield1 = []
    fitYield2 = []
    for i in range(0, input_data.shape[0]):
        input_dict1 = input.render(SowDay=(input_data['SowDay'].iloc[i] - 50)/50,
                                   SoilWC=(input_data['SoilWC'].iloc[i] - 15)/10,
                                   NRate=(input_data['NRate'].iloc[i])/200,
                                   Tillers=(input_data['TILLERS_SQ'].iloc[i] - 200)/800,
                                   Rain00=input_data['Rain00'].iloc[i]/150,
                                   Rain01=input_data['Rain01'].iloc[i]/150,
                                   Rain02=input_data['Rain02'].iloc[i]/150,
                                   Rain03=input_data['Rain03'].iloc[i]/150,
                                   Rain04=input_data['Rain04'].iloc[i]/150,
                                   Rain05=input_data['Rain05'].iloc[i]/150,
                                   Rain06=input_data['Rain06'].iloc[i]/150,
                                   Rain07=input_data['Rain07'].iloc[i]/150,
                                   Rain08=input_data['Rain08'].iloc[i]/150,
                                   Rain09=input_data['Rain09'].iloc[i]/150,
                                   Rad00=input_data['Rad00'].iloc[i]/500,
                                   Rad01=input_data['Rad01'].iloc[i]/500,
                                   Rad02=input_data['Rad02'].iloc[i]/500,
                                   Rad03=input_data['Rad03'].iloc[i]/500,
                                   Rad04=input_data['Rad04'].iloc[i]/500,
                                   Rad05=input_data['Rad05'].iloc[i]/500,
                                   Rad06=input_data['Rad06'].iloc[i]/500,
                                   Rad07=input_data['Rad07'].iloc[i]/500,
                                   Rad08=input_data['Rad08'].iloc[i]/500,
                                   Rad09=input_data['Rad09'].iloc[i]/500)
        input_dict1 = eval(input_dict1)
        input_value1 = np.concatenate((np.array([[1]]), np.array(list(input_dict1[0].values()))[:, np.newaxis]))

        input_dict2 = input.render(SowDay=(input_data2['SowDay'].iloc[i] - 50) / 50,
                                   SoilWC=(input_data2['SoilWC'].iloc[i] - 15) / 10,
                                   NRate=(input_data2['NRate'].iloc[i] + nrate) / 200,
                                   Tillers=(input_data2['TILLERS_SQ'].iloc[i] - 200) / 800,
                                   Rain00=input_data2['Rain00'].iloc[i] / 150,
                                   Rain01=input_data2['Rain01'].iloc[i] / 150,
                                   Rain02=input_data2['Rain02'].iloc[i] / 150,
                                   Rain03=input_data2['Rain03'].iloc[i] / 150,
                                   Rain04=input_data2['Rain04'].iloc[i] / 150,
                                   Rain05=input_data2['Rain05'].iloc[i] / 150,
                                   Rain06=input_data2['Rain06'].iloc[i] / 150,
                                   Rain07=input_data2['Rain07'].iloc[i] / 150,
                                   Rain08=input_data2['Rain08'].iloc[i] / 150,
                                   Rain09=input_data2['Rain09'].iloc[i] / 150,
                                   Rad00=input_data2['Rad00'].iloc[i] / 500,
                                   Rad01=input_data2['Rad01'].iloc[i] / 500,
                                   Rad02=input_data2['Rad02'].iloc[i] / 500,
                                   Rad03=input_data2['Rad03'].iloc[i] / 500,
                                   Rad04=input_data2['Rad04'].iloc[i] / 500,
                                   Rad05=input_data2['Rad05'].iloc[i] / 500,
                                   Rad06=input_data2['Rad06'].iloc[i] / 500,
                                   Rad07=input_data2['Rad07'].iloc[i] / 500,
                                   Rad08=input_data2['Rad08'].iloc[i] / 500,
                                   Rad09=input_data2['Rad09'].iloc[i] / 500)

        input_dict2 = eval(input_dict2)
        input_value2 = np.concatenate((np.array([[1]]), np.array(list(input_dict2[0].values()))[:, np.newaxis]))

        ## neuron network
        hidden_out1 = sigmoid(input_value1.transpose().dot(coeff_hidden))
        fitYield1.append(np.concatenate((np.array([[1]]), hidden_out1), axis=1).dot(coeff_output))
        hidden_out2 = sigmoid(input_value2.transpose().dot(coeff_hidden))
        fitYield2.append(np.concatenate((np.array([[1]]), hidden_out2), axis=1).dot(coeff_output))
    if nrate == 0:
        print ('haha')
    #### accuracy assessment and plot
    # tureYield = np.array(input_data['Yield'])[:, np.newaxis]
    tureYield = np.array(fitYield2)
    fitYield = np.array(fitYield1)
    fig = plt.figure(1, figsize=(8, 6), facecolor='#FFFFFF')
    plt.scatter(fitYield, tureYield, c='#fc8d59', s=15)
    regr_train = linear_model.LinearRegression()
    regr_train.fit(fitYield, tureYield)
    score_train[index] = regr_train.score(fitYield, tureYield)
    fittedline_train = regr_train.predict(fitYield)

plt.plot(fitYield, fittedline_train, linestyle='dashed', linewidth=1, c='#fc8d59')
plt.plot(range(-100, 101), score_train, linestyle='-', linewidth=1, c='#fc8d59')
ax = plt.gca()
labels = ax.get_xticklabels() + ax.get_yticklabels()
for label in labels:
    label.set_fontname('Times New Roman')
    label.set_style('italic')
    label.set_fontsize(16)
    label.set_weight('bold')

plt.legend(loc=4, facecolor='none', edgecolor='none',
           prop={
               'family': 'Times New Roman',
               # 'weight': 'bold',
               'size': 18
               # 'style': 'italic'
           })

# plt.text(0.05, 0.6, '$R^2$ = ' + str(round(score_train, 4)), fontdict={
#     'family': 'Times New Roman',
#     # 'weight': 'bold',
#     'size': 18
#     # 'style': 'italic'
# })
ax.set_xlabel('Change of N Rate ', font1)
ax.set_ylabel('$R^2$', font1)
plt.grid(True, alpha=1, color='w')
rect = ax.patch  # a Rectangle instance
# rect.set_facecolor([0.94, 0.94, 0.94])
# ax.axis([0, 0.7, -0.1, 0.7])
ax.axis([-110, 110, 0.98, 1.02])
plt.show()
