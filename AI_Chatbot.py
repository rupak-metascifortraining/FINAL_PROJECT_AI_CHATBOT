# QnA Bot - AI in Finance (Prototype)

import streamlit as st
from transformers import pipeline

# Title and Description
st.title("💬 QnA Bot - AI in Finance and Banking")
st.markdown("""
This is a prototype QnA bot designed to answer finance and banking-related questions using a pre-trained NLP model. 
It uses HuggingFace's transformers and is fine-tuned for extractive question answering.
""")

# Load Pretrained QA Pipeline (Publicly available SQuAD model)
st.info("Loading model...")
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
st.success("Model loaded successfully!")

# Comprehensive Financial and Banking Context
financial_context = """
A mutual fund is a professionally managed investment fund that pools money from many investors to purchase securities.
The repo rate is the interest rate at which a central bank lends money to commercial banks.
EPS stands for Earnings Per Share and is calculated by dividing net profit by the number of outstanding shares.
Inflation is the rate at which the general level of prices for goods and services is rising.
GDP stands for Gross Domestic Product and represents the total dollar value of all goods and services produced over a specific time period.
A stock represents a share in the ownership of a company and constitutes a claim on part of the company’s assets and earnings.
Bonds are fixed income instruments that represent a loan made by an investor to a borrower.
Asset allocation involves dividing an investment portfolio among different asset categories, such as stocks, bonds, and cash.
NAV (Net Asset Value) is the value per share of a mutual fund or an exchange-traded fund (ETF). It is calculated as the total value of assets minus total liabilities, divided by the number of outstanding shares.
Capital gain refers to the increase in a capital asset's value and is considered realized when the asset is sold.
A budget deficit occurs when expenses exceed revenue, indicating the government is spending more than it earns.
Credit score is a number representing the creditworthiness of a person, used by lenders to assess risk.
Derivatives are financial instruments whose value depends on the value of underlying assets such as stocks, bonds, or interest rates.
Monetary policy involves the processes by which a central bank controls the money supply to achieve specific goals.
Interest rate is the proportion of a loan that is charged as interest to the borrower.
Diversification is a risk management strategy that mixes a wide variety of investments within a portfolio.
Fiscal policy refers to the government's use of taxation and spending to influence the economy.
Banking services include savings accounts, checking accounts, fixed deposits, personal and business loans, credit and debit cards, and online banking.
A savings account is a deposit account held at a financial institution that provides interest while keeping funds accessible.
A current account is primarily for business purposes, offering frequent and unlimited transactions.
Online banking allows customers to conduct financial transactions via the internet, including transfers, bill payments, and account monitoring.
NEFT (National Electronic Funds Transfer) and RTGS (Real-Time Gross Settlement) are electronic payment systems used for interbank transfers in India.
Blockchain is a distributed ledger technology that underpins cryptocurrencies and allows secure and transparent record-keeping.
Cryptocurrency is a digital or virtual currency secured by cryptography, operating independently of a central bank.
An IPO (Initial Public Offering) is the process through which a private company offers shares to the public for the first time.
A balance sheet is a financial statement that summarizes a company’s assets, liabilities, and shareholders’ equity at a specific point in time.
Cash flow statements track the flow of cash in and out of a business over a given period.
An income statement shows a company’s revenues and expenses over a particular period, providing insight into profitability.
Forex (foreign exchange) market is a global decentralized market for trading currencies.
Microfinance is a type of banking service provided to unemployed or low-income individuals or groups who otherwise have no access to financial services.
"""

# User Input
question = st.text_input("Ask a finance or banking-related question:")

# Answer the Question
if question:
    with st.spinner("Searching for the answer..."):
        result = qa_pipeline({
            'question': question,
            'context': financial_context
        })
        st.success("Answer:")
        st.write(result['answer'])

# Footer
st.markdown("---")

