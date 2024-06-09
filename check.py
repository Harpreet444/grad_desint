import streamlit as st
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt


st.set_page_config(page_icon="🤖",page_title="Gradient descent algorithm")

st.title("Gradient descent algorithm")

st.write("Gradient descent is an optimization algorithm used in machine learning to minimize the cost function by iteratively updating the model's parameters. In the context of linear regression, it adjusts the slope m and intercept b to best fit the data.\nIn short, gradient descent uses the tangent (derivative) of the cost function to iteratively move the parameters towards the minimum cost, thereby optimizing the model.")

url = 'https://www.desmos.com/calculator/gmjeo0u1ge?embed'


x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

data = {'x':[1,2,3,4,5],'y':[5,7,9,11,13]}
data = pd.DataFrame(data)

html_code = f"""
<iframe src="{url}" width="500" height="500" style="border: 1px solid #ccc" frameborder=0></iframe>
"""
st.markdown(html_code, unsafe_allow_html=True)

st.subheader("Find the best fit line of given data using Gradient descent algorithm")

st.table(data)

st.write("Algorithm code:")
st.code("""
def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.001
    prev = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        cost = (1/n) + sum([val**2 for val in (y-y_predicted)])

        if math.isclose(cost,prev,rel_tol=1e-09):
            break

        print("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost,i))
        prev = cost
        """)

st.subheader("Scatter diaagram:")
fig, ax = plt.subplots()
ax.scatter(data.x,data.y, color='b', marker = '*')  # Experience vs. Salary
ax.set_xlabel('X')
ax.set_ylabel('Y')
st.pyplot(fig)

def gradient_descent(X, y, grad):
  m_curr = b_curr = 0
  iterations = 10000
  n = len(X)
  learning_rate = grad
  output_placeholder = st.empty()
  output_text = ""
  prev = 0


  for i in range(iterations):
      y_predicted = m_curr*x + b_curr
      cost = (1/n)*sum([val**2 for val in (y-y_predicted)])
      md = -(2/n)*sum(x*(y-y_predicted))          #derivative of m
      bd = -(2 / n) * sum(y - y_predicted)        #derivative of b
      m_curr = m_curr - learning_rate*md
      b_curr = b_curr - learning_rate * bd

      if prev == cost:
        st.subheader("Output:")
        st.code("""m = {}
b = {}
cost = {}
iteration = {} """.format(m_curr,b_curr,cost,i))
        
        fig, ax = plt.subplots()
        ax.scatter(data.x,data.y, color='b', marker = '*')  # Experience vs. Salary
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        y_pred_line = m_curr * x + b_curr
        ax.plot(x, y_pred_line, color='red')
        st.pyplot(fig)

        break

      prev = cost

grad = st.slider(label="Choose Learning rate",min_value=0.01,max_value=0.1,step=0.001)
st.write("learning rate = ",grad)
btn = st.button(label="Run Algo.")


if btn:
  gradient_descent(x, y, grad)