import math
import pandas as pd
import numpy as np
import scipy
from datetime import datetime, timedelta

from black_scholes_calculator import (
  blackScholes, optionDelta, optionGamma, optionTheta, optionVega, optionRho
)

import scipy.optimize

class Strategy:
  
  def __init__(self) -> None:
    self.capital : float = 100_000_000
    self.portfolio_value : float = 0

    self.start_date : datetime = datetime(2024, 1, 1)
    self.end_date : datetime = datetime(2024, 1, 3)
  
    self.options : pd.DataFrame = pd.read_csv("data/cleaned_options_data.csv")
    self.options["day"] = self.options["ts_recv"].apply(lambda x: x.split("T")[0])

    self.underlying = pd.read_csv("data/underlying_data_hour.csv")

    self.underlying["date"] = self.underlying["date"].apply(lambda x: x.split(" ")[0])
    self.underlying.columns = self.underlying.columns.str.lower()
    self.greek_bounds = {
            'delta': 10000,
            'gamma': 10000,
            'vega': 10000,
            'theta': 10000,
            'rho': 10000
        }
    self.greek_utilization = {greek: 0 for greek in self.greek_bounds}

  def parse_option_symbol(self, symbol):
    """
    example: SPX   240419C00800000
    """
    numbers : str = symbol.split(" ")[3]
    date : str = numbers[:6]
    date_yymmdd : str = "20" + date[0:2] + "-" + date[2:4] + "-" + date[4:6]
    action : str = numbers[6]
    strike_price : float = float(numbers[7:]) / 1000
    return [datetime.strptime(date_yymmdd, "%Y-%m-%d"), action, strike_price]
  
  #function takes in a dataframe row and returns float representing 
  def calc_weighted_mid_price(self, option_row) -> float:
    bid_px = option_row['bid_px_00']#need 0 to access first row and return float
    ask_px = option_row['ask_px_00']#need 0 to access first row and return float
    bid_sz = option_row['bid_sz_00']
    ask_sz = option_row['ask_sz_00']

    weighted_mid_price = ((bid_px * ask_sz) + (bid_sz * ask_px))/(ask_sz + bid_sz)

    mid_price = (bid_px + ask_px)/2 

    spread = ask_px - bid_px

    return (weighted_mid_price - mid_price)/spread #num between 0 and 1 representing trend of weighted mid price


  #calculates where stock will be on expiration date
  def future_price(self, asset_price, time_expire, risk_free_rate):
    return asset_price/np.exp(-risk_free_rate* time_expire)
  
  #find implied volatility from black sholes
  def implied_vol(self, option_price, asset_price, strike_price, time_expire, risk_free_rate, action):

    deflater = np.exp(-risk_free_rate* time_expire)
    undiscounted_option_price = option_price / deflater
    future = self.future_price(asset_price, time_expire, risk_free_rate)
    sigma_calc = self.find_iv(undiscounted_option_price,future, strike_price, time_expire, action)

    return sigma_calc

  @staticmethod
  def cdf(x):
    # moni's cdf 
    return 0.5*(1 + math.erf(x/math.sqrt(2)))
    # return 0.5*(1+scipy.special.erf(x/np.sqrt(2)))
    # return scipy.stats.norm.cdf(x)
    
  def find_sigma_with_scipy(self, option_price, S, K, T, r, action):
    option_type = 'c' if action == 'C' else 'p'
    
    def objective(sigma):
        return blackScholes(S, K, r, T, sigma, option_type) - option_price

    try:
        result = scipy.optimize.brentq(objective, 1e-10, 10, xtol=1e-15)
        return result
    except ValueError: 
        return scipy.optimize.minimize_scalar(lambda x: abs(objective(x)),
                                              bounds=(1e-10,10),
                                              method='bounded').x
    # obj_fn = lambda x: self.black_scholes(S, K, T, r, x, action) - option_price
    # dfn = lambda sigma: self.vega(S, K, T, r, sigma,action) *100
    # def objective(sigma):
    #   return Strategy.black_scholes(S, K, T, r, sigma, action) - option_price

    # try:
    #     # result = scipy.optimize.newton(func=obj_fn, x0=.8, fprime=dfn, tol=1e-5, maxiter=100)
    #     result = scipy.optimize.brentq(objective, 1e-10, 10, xtol=1e-15)
    #     return result
    # # except RuntimeError as e:
    # #     return .01
    # # if brntq fails, try a different method 
    # except ValueError: 
    #   return scipy.optimize.minimize_scalar(lambda x: abs(objective(x)),
    #                                         bounds=(1e-10,10),
    #                                         method='bounded').x
    
  #rudimentary program that performs newton raphson method of finding roots
  def find_sigma(self, option_price, S, K, T, r, action):
    max_iterations = 200
    precision = 1.0e-5
    
    sigma = 1
    option_type = 'c' if action == 'C' else 'p'

    for i in range(0, max_iterations):
        if sigma < -.1:
            return 0.0001

        price = blackScholes(S, K, r, T, sigma, option_type)
        vega = optionVega(S, K, r, T, sigma)

        if vega == 0:
            return 0.0001
        
        diff = option_price - price
        if (abs(diff) < precision):
            return sigma
        sigma = sigma + diff/(vega * 100)
    return sigma
  
    # max_iterations = 200
    # precision = 1.0e-5
    
    # #use closed form approx for sigma to prevent NaN errors
    # #link https://quant.stackexchange.com/questions/7761/a-simple-formula-for-calculating-implied-volatility?rq=1
    # #sigma = np.sqrt((option_price * 2 * np.pi)/(T * S))#anything under ~.75 for initial guess will over shoot and return NaN
    # sigma = 1
    # for i in range(0, max_iterations):

    #   #note that if options is priced cheaper than 0 volatility sigma will fail and become negative
    #   # we account for this by checking if sigma is 0 implying we are far from root/nonexistent 
    #   if sigma < -.1:
    #     return 0.0001 # just return very small sigma
      
    #   price = self.black_scholes(S, K, T, r, sigma, action)
    #   #print("Option price from sholes is  "+str(price) + " and volatility is " + str(sigma))
    #   vega = self.vega(S, K, T, r, sigma,action)

    #   #again since vega must always be positive if option is prices lower than therotical cheapest price
    #   #fix by return vega slightly above 0
    #   if vega == 0:
    #     return 0.0001 #small vega implies small sigma (for same reason)
      
    #   diff = option_price - price  # our root
    #   if (abs(diff) < precision):
    #       return sigma
    #   sigma = sigma + diff/(vega * 100) # f(x) / f'(x)
    # return sigma # value wasn't found, return best guess so far
  
  def getAllGreeks(self, option_price, asset_price, strike_price, time_expire, risk_free_rate, action):
    #print("PARAMETERS FOR FINDING THE GREEKS")
    #print([option_price, asset_price, strike_price, time_expire, risk_free_rate, action])
    
    #this sigma will be used throughout the other greek calculations
    #sigma = self.find_sigma(option_price, asset_price, strike_price, time_expire, risk_free_rate, action)

    #check performance boost with scipy optimize
    sigma = self.find_sigma_with_scipy(option_price, asset_price, strike_price, time_expire, risk_free_rate, action)
    option_type = 'c' if action == 'C' else 'p'

    #find all greeks
    delta = optionDelta(asset_price, strike_price, risk_free_rate, time_expire, sigma, option_type)
    gamma = optionGamma(asset_price, strike_price, risk_free_rate, time_expire, sigma)
    theta = optionTheta(asset_price, strike_price, risk_free_rate, time_expire, sigma, option_type)
    vega = optionVega(asset_price, strike_price, risk_free_rate, time_expire, sigma)
    rho = optionRho(asset_price, strike_price, risk_free_rate, time_expire, sigma, option_type)
    
    #return as dict to merge with options data
    return {'sigma': sigma, 'delta': delta, 'vega': vega, 'theta': theta, 'rho': rho, 'gamma': gamma}


  def new_order(self, date, option, action, sz):
    return {"datetime": date, "option_symbol": option, "action":action,"order_size":sz}
  
  #This function takes into account the strike price and the spx price to quote optimal bid/ask sizes
  # pass in available size 
  def order_size_wrt_spx(self, spx_price, strike_price, available_size):

    #the center of the normal distribution
    center = spx_price

    #independent variable is spread of distribution will leave at 20 dollars for now
    #arbitrarily chosen 
    std = 20

    #the quantile in the norm distribution we have
    quantile = scipy.stats.norm.cdf(strike_price, spx_price, std)

    # a ratio from 0 to 1 of how close we are to stock mean 1 being center 0 being very far away
    ratio = abs(.5 - quantile)/.5

    #another independent variable, (or future dependent variable on prices or greeks?)
    #arbitrarily chosen for now 
    stock_sz_multiplier = 500

    #calculate theoretical size
    theoretical_size = np.round(ratio * stock_sz_multiplier, 0)

    return min(theoretical_size, available_size)

  #function quotes a lower or higher option price depending on weighted mid price 
  def make_order(self, option_row, spx_price ):
    # this number will from 0 to 1, if close to 0 our iteration of bid/ask should either be higher or lower
    # our strategy for 1 should just be the opposite
    weighted_ratio = self.calc_weighted_mid_price(option_row)
    #print(weighted_ratio)
    
    symbolInfo = self.parse_option_symbol(option_row['symbol']) 


    #getting strike price
    strike_price = symbolInfo[2]

    if (weighted_ratio < 0.5): #if midprice closer to bid lots of buying pressure, so BUY
      available_size = option_row["ask_sz_00"]
      # order size will be min of g(x) and asking price
      size = self.order_size_wrt_spx(spx_price, strike_price, available_size)
      action = 'B'
      price = option_row["ask_px_00"]
    else: 
      available_size = option_row["bid_sz_00"]
      size = self.order_size_wrt_spx(spx_price, strike_price, available_size)
      action = 'S'
      price = option_row["bid_px_00"]

    #ensure we're not selling more than we own 
    if action == 'S':
      current_position = self.get_current_position(option_row['symbol'])
      size = min(size, current_position)
    
    # slippage is positive so rasises price by .1 percent of order size
    order = self.new_order(option_row["ts_recv"], option_row['symbol'], action, size)
    slippage = size * 0.001 
    price += slippage

    return [order, price]
  
  def get_current_position(self, option_symbol):
    return self.positions.get(option_symbol, 0)

  def update_position(self, option_symbol, size, action):
    if option_symbol not in self.positions:
        self.positions[option_symbol] = 0
    if action == 'B':
        self.positions[option_symbol] += size
    else:
        self.positions[option_symbol] -= size
  


  #find last trade to of option to get reliable option_price
  def get_last_option_price(self, option, date):
    option_dates = self.options[self.options['symbol'] == option] #filter to only include option info we want
    option_dates['ts_recv'] = option_dates['ts_recv'].str[:-4]
    option_dates['ts_recv'] = option_dates['ts_recv'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f'))
    #print(option_dates['ts_recv'] )
    lowerneighbour_ind = option_dates[option_dates['ts_recv'] <= date]['ts_recv']
    if lowerneighbour_ind.empty:
      return -64
    
    option = self.options.loc[lowerneighbour_ind.idxmax()]
    mid_price = (option['bid_px_00'] + option['ask_px_00'])/2 
    #print("THE MID PRICE")
    #print(mid_price)
    return mid_price # use weighted mid price as options price

  #updates individual orders with new greek information based on new spx and option prices
  def update_order_info(self, row, spx_price, date, risk_free_rate):

    #print("updating order " + str(row))
    #print(row) #only dates?
    symbol_info = self.parse_option_symbol(row['option_symbol'])
    strike_price = symbol_info[2]
    action = symbol_info[1]
    #print(date)
    date = date[:-4] # issues parsing time zone information without messy errors DO NOT TOUCH
    date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f')
    expire_date = symbol_info[0]
    time_expire = (expire_date - date).days/365 #need in a yearly basis

    option_price = self.get_last_option_price(row['option_symbol'], date)
    #symbol used to represent no prev assets
    if option_price == -64:
      #print("No option price found ")
      #also set greeks to 0
      return row

    #need option price which should just be most recent order
    greek_info = self.getAllGreeks(option_price, spx_price, strike_price, time_expire, risk_free_rate, action)

    row.update(greek_info)#update new greek info to row info

    return row
    

  # this function updates the portfolio with information from the new spx open price
  def update_portfolio(self, open_orders_with_greeks, spx_price, date, risk_free_rate):
    #print(open_orders_with_greeks)
    #every order now has new greek info (optimize later for repeated operations)
    return open_orders_with_greeks.apply(lambda row: self.update_order_info(row, spx_price, date, risk_free_rate),axis=1)

  #responsible for tracking net movements we have in options in portfolio
  def update_open_orders(self, new_order, open_orders, spx_price):
    #find option symbol in open_orders

    found_order_index = open_orders[open_orders['option_symbol'] == new_order['option_symbol']]

    #if empty order is not found just make new order
    if found_order_index.empty:
      new_order = self.update_order_info(new_order, spx_price, new_order['datetime'], risk_free_rate=.03)
      open_orders.loc[len(open_orders)] = new_order
      return open_orders
    else: #order found add new_order info to this
      #print("the dataframe is " + str(open_orders))
      #print("the index is " + str(found_order_index.iloc[0]) + " " + str(open_orders.size))
      found_order = found_order_index.iloc[0]
      #print('The found order is ' + str(found_order))
      #date will always be updated to new_order
      found_order['datetime'] = new_order['datetime']

      # if orders match action then add everything sizes together
      if found_order['action'] == new_order['action']:
        found_order['order_size'] += new_order['order_size']
      elif found_order['order_size'] == new_order['order_size']:
        # remove order since neutralized
        open_orders = open_orders[open_orders['option_symbol'] != found_order['option_symbol']]
        return open_orders #return since no new information needed
      elif found_order['order_size'] >= new_order['order_size']:
        found_order['order_size'] -= new_order['order_size']
      else: #since new order is greater and opposite it changes action
        found_order['action'] = new_order['action']
        found_order['order_size'] = new_order['order_size'] - found_order['order_size']
      return open_orders
      
      # #add found_order back to list
      # remaining = open_orders.loc[open_orders['option_symbol'] == found_order['option_symbol']]

      # found_order = self.update_order_info(found_order, spx_price, new_order['datetime'], risk_free_rate=.03)

      # # add index to dictionary
      # remaining = open_orders.loc[open_orders['option_symbol'] == found_order['option_symbol']]

      # dicts = remaining.to_dict()

      # found_order = found_order.set_index(remaining.index.tolist()[0]).T.to_dict()



      # #use list comprehension to the dict value
      # #remaining = {key: dict({remaining.index.tolist()[0],found_order[key]}) for key in found_order}
      
      # open_orders.loc[open_orders['option_symbol'] == found_order['option_symbol']] = found_order

    return open_orders

  #returns the orders that were sold in order to include draft_order
  def check_update_portfolio(self, draft_order, open_orders, spx_price):
    #if open orders are empty then cant sell anything
    if len(open_orders.index) == 0:
      return [[],open_orders]

    # for now our greek bounds will be centered at 0 but skewed bounds may produce interesting results
    #get the greeks to iterate over list
    greeks = open_orders.columns[-6:]
    
    #print("Print all open orders in portfolio " + str(open_orders))
    #print("The draft order is " + str(draft_order))
    
    #since we are editing the greeks we need to save the original open_orders list
    open_orders_original = open_orders.copy()

    delta_bound = 1000 # may need to be WAY higher (only one open order of 5000 is included)
    vega_bound = 1000
    theta_bound = 1000
    rho_bound = 1000
    gamma_bound = 1000
    sigma_bound = 1000

    #bounds may need to be average of portfolio? weighted sum may not be best

    sell_orders = []
    
    #hold all bounds here
    bounds = {'sigma':sigma_bound, 'delta': delta_bound,'vega': vega_bound, 'theta': theta_bound,'rho': rho_bound,'gamma': gamma_bound}
    bounds = pd.DataFrame(data=bounds, columns=greeks, index=[0])
    # where we will store the calc bounds from portfolio
    calc_bounds = bounds.copy() 

    #zero the data
    out_of_bounds = calc_bounds.copy() * 0

    for greek in greeks:
      #print(calc_bounds.columns)
      # save weighted average in columns for later analysis
      open_orders[greek] = open_orders['order_size'] * open_orders[greek]
      # get sum of weighted average of greeks to get total portfolio
      calc_bounds[greek] = np.sum(open_orders[greek])

      # greeks are out of bounds take note by how much
      if calc_bounds[greek].loc[0] > bounds[greek].loc[0]:
        out_of_bounds[greek] = calc_bounds[greek] - bounds[greek]
      elif calc_bounds[greek].loc[0] < -bounds[greek].loc[0]:
        out_of_bounds[greek] = calc_bounds[greek] + bounds[greek]

    #sort by largest bounds from pandas frame and start at largest element
    calc = calc_bounds.to_dict(orient='list')
    sorted_bounds = sorted(calc, key=lambda k: abs(calc[k][0]), reverse=True)
    
    # portfolio bounds all in check
    if len(sorted_bounds) == 0:
      return [[],open_orders_original]
    
    #find asset slightly above or equal to magnitude change needed
    #if not found get max, then rerun
    #first need to split by direction

    # this checks the summed bounds of the portfolio
    # print("CHECKING PORTFOLIO BOUNDS")
    # print(open_orders)



    for greek in sorted_bounds:

      #implies that greek passes bounderies now
      if abs(calc_bounds[greek].loc[0]) < bounds[greek].loc[0]:
        continue

      if  out_of_bounds[greek].loc[0] > 0:
        #find asset slightly under or equal to magnitude change needed
        #if not found get max opposite, then rerun
        while out_of_bounds[greek].loc[0] > 0 and len(open_orders) != 0:

          #need to modify open_orders to use original to preserve weighted average

          upper_neighbour_ind = open_orders[open_orders[greek] < -out_of_bounds[greek].loc[0]][greek]

          #if not found get new upper_neighbour_ind which is slightly less than greek needed to get back to bounds
          if upper_neighbour_ind.empty:
            upper_neighbour_ind = open_orders[open_orders[greek] >= -out_of_bounds[greek].loc[0]][greek]
            upper_neighbour_ind = upper_neighbour_ind.idxmin()
          else:
            upper_neighbour_ind = upper_neighbour_ind.idxmax()
          
          #print(open_orders.iloc[upper_neighbour_ind])

          upper_neighbour = open_orders.loc[upper_neighbour_ind]
          #flip action
          upper_neighbour['action'] = 'S' if upper_neighbour['action'] == 'B' else "B"
          
          #update all greek bounds
          for greek in greeks:
            calc_bounds[greek] -= upper_neighbour[greek]
            out_of_bounds[greek] -= upper_neighbour[greek]

          sell_orders.append(upper_neighbour)

          #change action to buy and update open orders

          #print("Figuring out selling open_orders " + str(open_orders))
          open_orders = self.update_open_orders(upper_neighbour, open_orders, spx_price)
        #return zero sell orders done and empty list means do not add buy order
        if len(open_orders) == 0:
          for greek in greeks:
            open_orders[greek] = open_orders[greek] / open_orders['order_size']
          return [[],open_orders_original]

      elif out_of_bounds[greek].loc[0] < 0:
        #find asset slightly above or equal to magnitude change needed
        #if not found get max opposite, then rerun

        #case where we want to restore if not possible 
        #if list becomes 0 not possible to fix list 
        while out_of_bounds[greek].loc[0] < 0 and len(open_orders) != 0:
          upper_neighbour_ind = open_orders[open_orders[greek] > -out_of_bounds[greek].loc[0]][greek]

          #if not found get new upper_neighbour_ind
          if upper_neighbour_ind.empty:
            upper_neighbour_ind = open_orders[open_orders[greek] <= -out_of_bounds[greek].loc[0]][greek].idxmax()
          else:
            upper_neighbour_ind = upper_neighbour_ind.idxmin()

          upper_neighbour = open_orders.loc[upper_neighbour_ind]
          #flip action
          upper_neighbour['action'] = 'S' if upper_neighbour['action'] == 'B' else "B"

          #update all greek bounds
          for greek in greeks:
            calc_bounds[greek] += upper_neighbour[greek]
            out_of_bounds[greek] += upper_neighbour[greek]

          sell_orders.append(upper_neighbour)

          #change action to buy and update open orders
          open_orders = self.update_open_orders(upper_neighbour, open_orders, spx_price)
        #return zero sell orders done and empty list means do not add buy order
        if len(open_orders) == 0:
          for greek in greeks:
            open_orders[greek] = open_orders[greek] / open_orders['order_size']
          return [[],open_orders_original]
        
    #reset open_order weighted average
        
    sell_orders = pd.DataFrame(sell_orders)

    #print("the sell order is " + sell_orders)
    for greek in greeks:
      open_orders[greek] = open_orders[greek] / open_orders['order_size']
    
    if not sell_orders.empty:
      for greek in greeks:
        sell_orders[greek] = sell_orders[greek] / sell_orders['order_size']


      
    return[sell_orders,open_orders]

  #get the margins needed according to case spec: spx price is opening price
  def get_margin(self, draft_order, spx_price, premium):
    #get symbol info to see if call or put order
    symbol_info = self.parse_option_symbol(draft_order['option_symbol'])
    put_call = symbol_info[1]
    strike_price = symbol_info[2]
    
    #must include the 100x multiplier
    if put_call == 'P':
      return (premium + .1 * spx_price )*100 
    else:
      return (premium + .1 * strike_price )*100 

  # note that price can either be the ask price (if buying) or bid price (if selling)
  def update_capital_value(self,order, price, spx_price):

    order_size = order['order_size']
    
    symbol_info = self.parse_option_symbol(order['option_symbol'])
    strike_price = symbol_info[2]

    if order["action"] == "B":
      options_cost: float = order_size * price + 0.1 * strike_price
      margin: float = (price + 0.1 * strike_price) * order_size
      self.capital -= options_cost + 0.5
      self.portfolio_value += options_cost
    else:
      #underlying price is not open price but for now yes
      sold_stock_cost: float = order_size * 100 * spx_price
      #should be open price for margin
      margin : float = 100 * order_size * (price + 0.1 * spx_price)
      if (self.capital + order_size * price + 0.1 * strike_price) > margin and (self.capital + order_size * price + 0.1 * strike_price - sold_stock_cost + 0.5) > 0:
        self.capital += order_size * price + 0.1 * strike_price
        self.capital -= sold_stock_cost + 0.5
        self.portfolio_value += order_size * 100 * spx_price


  #responsible for selling assets to get back within our bounds 
  #reduce_portfolio()
  #check that greeks are within 80 percent of bounds, 
    

  # we want all orders to also include their respective greeks and we will update this every incrementing day?

  # def generate_orders(self) -> pd.DataFrame:

  #   #initialize positions dictionary if not already done 
  #   if not hasattr(self, 'positions'):
  #     self.positions = {}
    
  #   risk_free_rate = .03 #from case specs

  #   #basic outline looping day by day
  #   current_date = self.start_date
  #   delta = timedelta(days=1)

  #   # two seperately managed lists, orders with just pure order information
  #   # and open_orders which holds the net positions held in our portfolio to reduce unneccessary computation of greeks

  #   orders = pd.DataFrame(columns=['datetime','option_symbol','action','order_size'])
  #   open_orders = pd.DataFrame(columns=['datetime','option_symbol','action','order_size','sigma', 'delta', 'vega', 'theta','rho','gamma'])

  #   # order statistics 
  #   total_options_considered = 0 
  #   total_orders_generated = 0 
  #   total_portfolio_updates = 0
    
  #   while(current_date < self.end_date):
  #     #debugging print statements 
  #     print(f"Processing date: {current_date}")
  #     #get the same day for the spx

  #     spx_day = self.underlying[self.underlying["date"] == current_date.strftime("%Y-%m-%d")]
      
  #     #check spx data not empty
  #     if spx_day.empty: 
  #       print(f"No SPX data for {current_date}, skipping.")
  #       current_date += delta
  #       continue
    
  #     #get spx price for day 
  #     spx_price = spx_day['open'].iloc[0]
  #     print(f"SPX price for {current_date}: {spx_price}")

  #     if not open_orders.empty:
  #       open_orders = self.update_portfolio(open_orders, spx_price, current_date.strftime('%Y-%m-%dT%H:%M:%S.%f'), risk_free_rate)
  #       total_portfolio_updates += 1
  #       print(f"Updated portfolio. Current open orders: {len(open_orders)}")

  #     day_options = self.options[self.options["day"] == current_date.strftime("%Y-%m-%d")]
  #     print(f"Number of options to process for {current_date}: {len(day_options)}")

  #     for _, option_order in day_options.iterrows():
  #       total_options_considered += 1

  #       order_and_price = self.make_order(option_order, spx_price)
  #       draft_order, price = order_and_price

  #       print(f"Considering order: {draft_order}")
       
  #       # Check if we have enough capital and margin
  #       margin = self.get_margin(draft_order, spx_price, price)
  #       if margin + transaction_cost > self.capital:
  #           continue
        
  #       # Update position and capital
  #       self.update_position(draft_order['option_symbol'], draft_order['order_size'], draft_order['action'])
  #       self.update_capital_value(draft_order, price, spx_price)

  #       # Add to orders
  #       orders.loc[len(orders)] = draft_order
  #       transaction_cost = 0.5

  #       if len(open_orders) == 0 and margin + transaction_cost > self.capital:
  #         print("Insufficient capital for this order, skipping.")
  #         continue

  #       order_package = self.check_update_portfolio(draft_order, open_orders, spx_price)
  #       sell_orders, open_orders = order_package

  #       for sell_order in sell_orders.iterrows():
  #         option = sell_order[1]
  #         date = datetime.strptime(option['datatime'][:-4], '%Y-%m-%dT%H:%M:%S.%f')
  #         sell_price = self.get_last_option_price(option['option_symbol'], date)
  #         self.update_capital_value(option, sell_price, spx_price)
  #         orders.loc[len(orders)] = option
  #         total_orders_generated += 1
  #         print(f"Generated sell order: {option}")
        
  #       if open_orders.empty and margin + transaction_cost > self.capital:
  #         print("insufficient captial after portfolio update, skipping")
  #         continue

  #       self.update_capital_value(draft_order, price, spx_price)
  #       orders.loc[len(orders)] = dict(list(draft_order, open_orders, spx_price))
  #       total_orders_generated += 1
  #       print(f"Generated buy order: {draft_order}")
      
  #     print(f"Finished processing {current_date}. Current capitial: {self.capital}, Portfolio value: {self.portfolio_value}")
  #     current_date += delta

  #   print(f"Strategy complete. Total options considered: {total_options_considered}")
  #   print(f"Total orders generated: {total_orders_generated}")
  #   print(f"Total portfolio updates: {total_portfolio_updates}")

  #   self.orders = orders
  #   return orders
  
  def generate_orders(self) -> pd.DataFrame:
        risk_free_rate = 0.03
        current_date = self.start_date
        delta = timedelta(days=1)

        orders = pd.DataFrame(columns=['datetime', 'option_symbol', 'action', 'order_size'])
        open_orders = pd.DataFrame(columns=['datetime', 'option_symbol', 'action', 'order_size', 'sigma', 'delta', 'vega', 'theta', 'rho', 'gamma'])

        total_options_considered = 0
        total_orders_generated = 0
        total_portfolio_updates = 0
        transaction_cost = 0.5

        while current_date < self.end_date:
            print(f"Processing date: {current_date}")
            
            spx_day = self.underlying[self.underlying["date"] == current_date.strftime("%Y-%m-%d")]
            
            if spx_day.empty:
                print(f"No SPX data for {current_date}, skipping.")
                current_date += delta
                continue
            
            spx_price = spx_day['open'].iloc[0]
            print(f"SPX price for {current_date}: {spx_price}")

            if not open_orders.empty:
                open_orders = self.update_portfolio(open_orders, spx_price, current_date.strftime('%Y-%m-%dT%H:%M:%S.%f'), risk_free_rate)
                self.update_greek_utilization(open_orders)
                total_portfolio_updates += 1
                print(f"Updated portfolio. Current open orders: {len(open_orders)}")

            day_options = self.options[self.options["day"] == current_date.strftime("%Y-%m-%d")]
            print(f"Number of options to process for {current_date}: {len(day_options)}")

            for _, option_order in day_options.iterrows():
                total_options_considered += 1

                order_and_price = self.make_order(option_order, spx_price)
                draft_order, price = order_and_price

                print(f"Considering order: {draft_order}")
                
                # Check if we have enough capital and margin
                margin = self.get_margin(draft_order, spx_price, price)

                if margin + transaction_cost > self.capital:
                    print("Insufficient capital for this order, skipping.")
                    continue
                
                # Check Greeks
                if not self.check_greek_limits(draft_order, spx_price, risk_free_rate):
                    print("Order exceeds Greek limits, attempting to rebalance portfolio.")
                    sell_orders = self.rebalance_portfolio(draft_order, open_orders, spx_price, risk_free_rate)
                    
                    for sell_order in sell_orders.iterrows():
                        option = sell_order[1]
                        date = datetime.strptime(option['datetime'][:-4], '%Y-%m-%dT%H:%M:%S.%f')
                        sell_price = self.get_last_option_price(option['option_symbol'], date)
                        self.update_capital_value(option, sell_price, spx_price)
                        orders = orders.append(option, ignore_index=True)
                        total_orders_generated += 1
                        print(f"Generated sell order: {option}")
                    
                    if not self.check_greek_limits(draft_order, spx_price, risk_free_rate):
                        print("Unable to rebalance portfolio within Greek limits, skipping order.")
                        continue

                # Update position and capital
                self.update_position(draft_order['option_symbol'], draft_order['order_size'], draft_order['action'])
                self.update_capital_value(draft_order, price, spx_price)

                # Add to orders
                orders = orders.append(draft_order, ignore_index=True)
                open_orders = self.update_open_orders(draft_order, open_orders, spx_price)
                self.update_greek_utilization(open_orders)
                total_orders_generated += 1
                print(f"Generated buy order: {draft_order}")
            
            print(f"Finished processing {current_date}. Current capital: {self.capital}, Portfolio value: {self.portfolio_value}")
            current_date += delta

        print(f"Strategy complete. Total options considered: {total_options_considered}")
        print(f"Total orders generated: {total_orders_generated}")
        print(f"Total portfolio updates: {total_portfolio_updates}")

        self.orders = orders
        return orders

  def check_greek_limits(self, draft_order, spx_price, risk_free_rate):
      symbol_info = self.parse_option_symbol(draft_order['option_symbol'])
      strike_price = symbol_info[2]
      action = symbol_info[1]
      expiry_date = symbol_info[0]
      time_to_expiry = (expiry_date - datetime.now()).days / 365

      option_price = self.get_last_option_price(draft_order['option_symbol'], datetime.now())
      if option_price == -64:
          return False

      sigma = self.implied_vol(option_price, spx_price, strike_price, time_to_expiry, risk_free_rate, action)
      
      greeks = {
          'delta': optionDelta(spx_price, strike_price, risk_free_rate, time_to_expiry, sigma, action),
          'gamma': optionGamma(spx_price, strike_price, risk_free_rate, time_to_expiry, sigma),
          'vega': optionVega(spx_price, strike_price, risk_free_rate, time_to_expiry, sigma),
          'theta': optionTheta(spx_price, strike_price, risk_free_rate, time_to_expiry, sigma, action),
          'rho': optionRho(spx_price, strike_price, risk_free_rate, time_to_expiry, sigma, action)
      }

      for greek, value in greeks.items():
          if abs(self.greek_utilization[greek] + value * draft_order['order_size']) > self.greek_bounds[greek]:
              return False
      return True

  def update_greek_utilization(self, open_orders):
      self.greek_utilization = {greek: 0 for greek in self.greek_bounds}
      for _, order in open_orders.iterrows():
          for greek in self.greek_bounds:
              self.greek_utilization[greek] += order[greek] * order['order_size']

  def rebalance_portfolio(self, draft_order, open_orders, spx_price, risk_free_rate):
      sell_orders = pd.DataFrame(columns=open_orders.columns)
      
      while not self.check_greek_limits(draft_order, spx_price, risk_free_rate) and not open_orders.empty:
          # Find the order that contributes most to the Greek we're trying to reduce
          max_contribution = 0
          max_greek = ''
          max_order = None
          
          for _, order in open_orders.iterrows():
              for greek in self.greek_bounds:
                  contribution = abs(order[greek] * order['order_size'])
                  if contribution > max_contribution:
                      max_contribution = contribution
                      max_greek = greek
                      max_order = order
          
          if max_order is None:
              break
          
          # Sell the order that contributes most to the Greek we're trying to reduce
          sell_order = max_order.copy()
          sell_order['action'] = 'S' if sell_order['action'] == 'B' else 'B'
          sell_orders = sell_orders.append(sell_order, ignore_index=True)
          
          # Remove the sold order from open_orders
          open_orders = open_orders[open_orders.index != max_order.name]
          
          # Update Greek utilization
          self.update_greek_utilization(open_orders)
      
      return sell_orders


    ##### BEGIN OLD GENERATE ORDERS FUNCTION ######
      # #use iloc to get float value
      # spx_price = next(iter(spx_day['open']), 'no match')

      # if spx_price == 'no match':
      #   current_date += delta
      #   continue

      # #update portfolio as sigma will not change throughout the day (freshest info)
      # open_orders = self.update_portfolio(open_orders, spx_price, current_date.strftime('%Y-%m-%dT%H:%M:%S.%f'), risk_free_rate)

      # #now get all options that were sold that day
      # day_options = self.options[self.options["day"] == current_date.strftime("%Y-%m-%d")]

      # #start looping through these options acting like real time market 
      # # 1) will create draft order 2) check addition to portfolio 3) if cant add to portfolio sell assets then add

      # print("Date: " + str(current_date)) #for sanity purposes

      # start_index = day_options.index[0]

      # for index, option_order in day_options.iterrows():
        
      #   #print(" PRINTING ALL OPEN ORDERS ") 
      #   #print(open_orders)
      #   if index % 1000 == 0: 
      #     print("Percent done with day %" + str(100* (- start_index + index)/float(len(day_options))))

      #   #print("Working with order " + str(option_order))

      #   order_and_price = self.make_order(option_order, spx_price)

      #   draft_order = order_and_price[0]

      #   if draft_order['option_symbol'] == 'SPX   241220P07000000':
      #     print("break point here")

      #   price = order_and_price[1]

      #   margin = self.get_margin(draft_order, spx_price, price)

      #   #think one order is one contract
      #   transaction_cost = 0.5

      #   #if portfolio empty, check if order can be made
      #   #only scenario where order will not be made
      #   if len(open_orders.index) == 0 and margin + transaction_cost> self.capital:
      #     continue
      #   elif len(open_orders.index) == 0: #when list is empty just add order
      #     #print("Number of orders " + str(draft_order))
      #     open_orders = self.update_open_orders(draft_order,open_orders, spx_price)
      #     self.update_capital_value(draft_order, price, spx_price)
      #     orders.loc[len(orders)] = dict(list(draft_order.items())[:4])
      #     continue
      #   #if options symbol in portfolio is empty and 
        
      #   #program will always choose to try to make order and sell portfolio assets as needed
      #   order_package = self.check_update_portfolio(draft_order, open_orders, spx_price)


      #   # do not want to update any values for this stock if list is null
      #   if len(order_package[1]) == 0:
      #     continue
      #   sell_orders = order_package[0]
      #   open_orders = order_package[1]

      #   #print("These are sell orders " + str(len(sell_orders)))

      #   for i in range(0, len(sell_orders)):

      #     option = sell_orders.iloc[i]

      #     date = option['datetime'][:-4] # issues parsing time zone information without messy errors DO NOT TOUCH
      #     date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f')
      #     #need to find the price of selling the stock from our portfolio
      #     sell_price = self.get_last_option_price(option['option_symbol'], date)
      #     self.update_capital_value(option, sell_price, spx_price)
      #     orders.loc[len(orders)] = option

      #   #check case where after cleaning portfolio margin still isnt reached
      #   if open_orders.empty and margin + transaction_cost> self.capital:
      #     continue

      #   # passed all condtions add to order and modify portfolio and captial values accordingly 
      #   self.update_capital_value(draft_order, price, spx_price)
      #   orders.loc[len(orders)] = dict(list(draft_order.items())[:4])
      #   open_orders = self.update_open_orders(draft_order,open_orders, spx_price)
        

      # current_date += delta #increment to next day

    #check all incoming orders and calculate weighted mid price and create potential order

    #check the greeks of the potential order
  
    #if greeks pass an 80 percent bound, with new order, then 2 options
    #a) add order and remove (least amount of big options or large amount of small options) to restore portfolio
    #b) do not add order (bad however since no old stocks would be sold)

    #update portfolio every day increment (since using S&P open price for greeks)


    #considerations
      # update capital and portfolio bounds to make sure not negative and stuff
      # account for slippage in transaction, and margins needed 
    

    #return all order history
    #
    # self.orders = orders
    # return orders
    # pass
