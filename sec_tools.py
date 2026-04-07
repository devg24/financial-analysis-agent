import requests
import pandas as pd
from langchain_core.tools import tool
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field



USER_AGENT = "Dev Goyal devgoyal9031@gmail.com" # Update this to your real email!
HEADERS = {"User-Agent": USER_AGENT}

def get_cik_from_ticker(ticker: str) -> str:
    ticker = ticker.upper()
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    for _, company_info in data.items():
        if company_info['ticker'] == ticker:
            return str(company_info['cik_str']).zfill(10)
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
# 1. Define the strict Pydantic Schema
class XBRLConceptInput(BaseModel):
    ticker: str = Field(
        ..., 
        description="The official uppercase ticker symbol (e.g., AAPL)."
    )
    concept: Literal[
        "Revenues", 
        "NetIncomeLoss", 
        "Assets", 
        "Liabilities",
        "GrossProfit",
        "OperatingIncomeLoss",
        "AssetsCurrent",
        "LiabilitiesCurrent",
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "EntityCommonStockSharesOutstanding"
    ] = Field(
        ..., 
        description="You MUST select the exact SEC XBRL concept from this list that best matches the user's request."
    )

# 2. Bind the schema to the tool
@tool(args_schema=XBRLConceptInput)
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
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
        
        print(f"[System: Fetching latest {concept} for {ticker}...]")
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        if "USD" not in data.get("units", {}):
            return f"No USD data found for {concept}."
            
        # 1. Convert to DataFrame
        df = pd.DataFrame(data["units"]["USD"])
        
        # 2. Convert date strings to datetime objects
        df['end'] = pd.to_datetime(df['end'])
        df['filed'] = pd.to_datetime(df['filed'])
        
        # 3. Filter for standard filings to avoid "preliminary" noise
        df = df[df['form'].isin(['10-Q', '10-K', '10-K/A', '10-Q/A'])]
        
        # 4. CRITICAL: Deduplicate. 
        # If the same period ('end') is reported multiple times, take the most recently filed one.
        df = df.sort_values(by=['end', 'filed'], ascending=[False, False])
        df = df.drop_duplicates(subset=['end'])
        
        # 5. Filter for the last 2 years
        current_year = datetime.now().year
        df = df[df['end'].dt.year >= (current_year - 2)]
        
        # 6. Take top 4 most recent periods
        df = df.head(4)
        
        if df.empty:
            return f"No recent (2024-2026) {concept} data available for {ticker}."
        
        summary = f"Latest official {concept} data for {ticker}:\n"
        for _, row in df.iterrows():
            formatted_val = f"${int(row['val']):,}"
            date_str = row['end'].strftime('%Y-%m-%d')
            summary += f"- Period End: {date_str} (Filed: {row['filed'].strftime('%Y-%m-%d')}): {formatted_val}\n"
            
        return summary

    except Exception as e:
        return f"Error fetching XBRL data: {str(e)}"

# Quick test block for the new function
if __name__ == "__main__":
    test_ticker = "MSFT"
    
    # Test 1: URL fetcher
    try:
        url = get_latest_10k_url(test_ticker)
        print(f"\n10-K URL: {url}")
    except Exception as e:
        print(f"URL Fetch Failed: {e}")
        
    # Test 2: XBRL fetcher
    test_concept = "NetIncomeLoss"
    print(get_company_concept_xbrl.invoke({"ticker": test_ticker, "concept": test_concept}))