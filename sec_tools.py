import requests
import json
import re
import pandas as pd
from langchain_core.tools import tool


USER_AGENT = "Dev Goyal devgoyal9031@gmail.com" # Update this to your real email!
HEADERS = {"User-Agent": USER_AGENT}

def get_cik_from_ticker(ticker: str) -> str:
    """Translates a ticker symbol to a 10-digit padded CIK. We need the CIK to query the SEC's EDGAR system for filings."""
    ticker = ticker.upper()
    url = "https://www.sec.gov/files/company_tickers.json"
    
    print(f"[System: Fetching CIK for {ticker} from SEC...]")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    
    data = response.json()
    
    # Iterate through the SEC ticker dictionary
    for _, company_info in data.items():
        if company_info['ticker'] == ticker:
            cik = str(company_info['cik_str'])
            # The SEC submissions API requires the CIK to be padded to 10 digits with leading zeros
            return cik.zfill(10)
            
    raise ValueError(f"Ticker {ticker} not found in SEC database.")

def get_latest_10k_url(ticker: str) -> str:
    """Finds the URL for the most recent 10-K filing for a given ticker."""
    try:
        cik = get_cik_from_ticker(ticker)
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        
        print(f"[System: Fetching filing history for CIK {cik}...]")
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        filings = response.json()['filings']['recent']
        
        # Search for the most recent 10-K
        for i, form in enumerate(filings['form']):
            if form == '10-K':
                accession_number = filings['accessionNumber'][i]
                # The SEC URL format removes dashes from the accession number
                accession_no_dashes = accession_number.replace('-', '')
                
                # Construct the final document URL
                document_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_no_dashes}/{accession_number}.txt"
                return document_url
                
        return f"No 10-K found for {ticker}."
        
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_company_concept_xbrl(ticker: str, concept: str) -> str:
    """
    Fetches official SEC accounting metrics for a company across recent quarters.
    CRITICAL INSTRUCTIONS:
    1. 'ticker': Must be the official uppercase ticker symbol (e.g., AAPL).
    2. 'concept': You MUST use one of these exact SEC XBRL concepts (case-sensitive):
       -- Core Size --
       - 'Revenues' (Total Revenue / Sales)
       - 'NetIncomeLoss' (Net Income / Profit)
       - 'Assets' (Total Assets)
       - 'Liabilities' (Total Liabilities)
       
       -- Margins & Liquidity --
       - 'GrossProfit' (Revenue minus Cost of Goods Sold)
       - 'OperatingIncomeLoss' (Operating Income)
       - 'AssetsCurrent' (Short-term assets like cash/inventory)
       - 'LiabilitiesCurrent' (Short-term debt)
       
       -- Cash Flow & Valuation --
       - 'NetCashProvidedByUsedInOperatingActivities' (Operating Cash Flow)
       - 'PaymentsToAcquirePropertyPlantAndEquipment' (Capital Expenditures / CapEx)
       - 'EntityCommonStockSharesOutstanding' (Total shares outstanding)
       
    Do not guess concepts. Only use the exact strings listed above.
    """
    try:
        cik = get_cik_from_ticker(ticker)
        
        # Using the company-concept API endpoint you discovered
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
        
        print(f"[System: Fetching XBRL data for {concept} from SEC...]")
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        data = response.json()
        
        # XBRL data can be messy (reported in different currencies). We'll grab the USD array.
        if "USD" not in data.get("units", {}):
            return f"No USD data found for {concept}."
            
        usd_data = data["units"]["USD"]
        
        # Load into a Pandas DataFrame to easily filter and sort
        df = pd.DataFrame(usd_data)
        
        # Filter for only standard quarterly (10-Q) or annual (10-K) data (usually ~90 or 365 days)
        # We also drop duplicates in case a company amended a filing
        df = df[df['form'].isin(['10-Q', '10-K'])]
        df = df.drop_duplicates(subset=['end'])
        
        # Sort by date and grab the 4 most recent periods
        df['end'] = pd.to_datetime(df['end'])
        df = df.sort_values(by='end', ascending=False).head(4)
        
        # Format the output for the LLM
        summary = f"Recent {concept} data for {ticker}:\n"
        for _, row in df.iterrows():
            # Format the large numbers nicely (e.g., $1,000,000)
            formatted_val = f"${int(row['val']):,}"
            date_str = row['end'].strftime('%Y-%m-%d')
            summary += f"- {date_str} (Form {row['form']}): {formatted_val}\n"
            
        return summary

    except Exception as e:
        return f"Error fetching XBRL data: {str(e)}"

# Quick test block for the new function
if __name__ == "__main__":
    test_ticker = "AAPL"
    
    # Test 1: URL fetcher
    try:
        url = get_latest_10k_url(test_ticker)
        print(f"\n10-K URL: {url}")
    except Exception as e:
        print(f"URL Fetch Failed: {e}")
        
    # Test 2: XBRL fetcher
    try:
        xbrl_data = get_company_concept_xbrl(test_ticker, "NetIncomeLoss")
        print("\n" + xbrl_data)
    except Exception as e:
        print(f"XBRL Fetch Failed: {e}")