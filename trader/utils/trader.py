import ccxt
import os
import pandas as pd
import importlib
import inspect
import datetime

class Trader(object):
    def __init__(self, tradeConfig, apiKey, apiSecret):
        """ Account Initialize """
        self.tradeConfig = tradeConfig
        if self.tradeConfig['type']=='spot':
            self.exchange = ccxt.binance({
                'apiKey': apiKey,
                'secret': apiSecret,
                'urls': {
                    'api': 'https://testnet.binance.vision/api'  # 手動設置為 Binance Spot 測試網
                },
                'option':{
                    'defaultType':'future' #spot 現貨 future:期貨
                }
            })
            self.exchange.set_sandbox_mode(True)  # 啟用測試模式
        else:
            self.exchange = ccxt.binance({
                'apiKey': apiKey,
                'secret': apiSecret,
                'option':{
                    'defaultType':'future' #spot 現貨 future:期貨
                }
            })
    def run(self):
        if not time_match(self.timeframe):
            print(f"Not math the timeframe:{datetime.datetime.now()}")
            return
        for symbol,signal in self.signals.items():

            """ Buy side: Initialize Order Informatin """

            if signal == 1:
                order_info = self.buy_order_info(symbol) 
                if order_info!=None:
                    self.buy()

            """ Sell side: Initialize Order Informatin """
            unselled_order = self.get_unselled_orders(symbol)
            if len(unselled_order)>0:
                for order in unselled_order:
                    # 執行賣單
                    if signal == -1:
                        self.sell(order)
                        pass
                    # 檢查止盈止損
                    elif signal == 0:
                        if self.upl_ratio(order) > 0.2 or self.upl_ratio<-0.1:
                            self.sell(order)
                        pass
            else:
                continue
    def upl_ratio(self, order_info) -> float:
        current_price = get_current_price(order_info['symbol'])
        initial_cost = order_info['cost']
        current_value = current_price * order_info['amount']
        upl = (current_value - initial_cost) / initial_cost
        return float(upl)

    def calculate_position(self,total_balance):
        #頭寸大小 = 賬戶大小*賬戶風險(1%)/失效點(10%)
        position = (total_balance * 0.01) / 0.1
        return position

    def get_balance(self):
        balance = self.exchange.fetch_balance()
        return float(balance['total']['USDT'])
    
    def buy_order_info(self,symbol):
        remain_balance = self.get_balance()
        spend = self.calculate_position(remain_balance)
        if spend<5:                                         #Minimun Order in Binance can't less than 5 USDT
            print("Remaining Balance isn't Sufficient.")
            return None
        commission_paid = spend * 0.001                     #Binance commission for general user:0.1%
        net_spend =spend-commission_paid
        amount = net_spend/get_current_price(symbol)
        order_info ={
            'symbol':symbol,
            'type':'market',
            'side':'buy',
            'amount':amount
        }
        return order_info
    def sell(self,order_info):
        try:
            order = self.exchange.create_order(
                symbol=order_info['symbol'],
                type='market',      # default市價單
                side='sell',
                amount=order_info['amount']   # 計算出的 BTC 數量
            )
            update_fields = {
                'sell_status': True,
                'sell_orderID': order.get('id'),            
                'sell_price': order.get('price'),           
                'sell_cost': order.get('cost'),            
                'sell_time': order.get('datetime')      
            }

            """ Output """
            message =f"Sell Order created: {update_fields['sell_time']}\nsymbol:{order_info['symbol']}\nside:sell\nprice:{order.get('price')}\ncost:{order.get('cost')}"
            print(message)
            message_dc = f"""``` 
                Order created: {update_fields['sell_time']}
                symbol: {order_info['symbol']}
                side: sell
                price: {order.get('price')}
                cost: {order.get('cost')}
                ```"""
            self.discord.send_message(message_dc)


            """ Storage """
            try:
                self.database.collection.update_one(
                    {'orderID':order_info['orderID']},
                    {'$set': update_fields}
                )

            except:
                print("Update Sell Order failed.")

        except ccxt.BaseError as e:
            print(f"Error creating order: {e}")
        
    def buy(self,order_info):
        try:
            order = self.exchange.create_order(
                symbol=order_info['symbol'],
                type='market',      # default市價單
                side='buy',
                amount=order_info['amount']   # 計算出的 BTC 數量
            )
            storage_info ={
                'symbol': order.get('symbol'),         #貨幣對
                'orderID': order.get('id'),            #訂單唯一ID
                'type': order.get('type'),             #市價單 or 限價單
                'side': order.get('side'),             #buy or sell
                'price': order.get('price'),           #成交價格(USDT)
                'amount': order.get('amount'),         #買入數量
                'cost': order.get('cost'),             #花費(USDT)
                'status': order.get('status'),         
                'datetime': order.get('datetime'),
                'info': order.get('info'),             #包含詳細的訂單訊息
                'sell_status':False,                    #完成的訂單賣出與否: True: 已賣出 False: 尚未賣出
                'sell_orderID':'',
                'sell_price':'',
                'sell_cost':'',
                'sell_time':''
            }
            # Status:訂單的執行進度 #
            # open:於活躍狀態，尚未完全執行，仍在訂單簿中等待成交。
            # closed: 全部成交
            # canceled:在完全執行之前被取消
            # expired: 因交易所的時間限制或條件而過期
            # partially_filled: 已部分成交，但仍未完全執行，剩餘部分仍掛在訂單簿中等待成交
            # rejected: 未成功進入訂單簿

            """ Output """
            message =f"Order created: {storage_info['datetime']}\nsymbol:{storage_info['symbol']}\nside:buy\nprice:{storage_info['price']}\ncost:{storage_info['cost']}\nstatus:{storage_info['status']}"
            print(message)
            message_dc = f"""``` 
                Order created: {storage_info['datetime']}
                symbol: {storage_info['symbol']}
                side: buy
                price: {storage_info['price']}
                cost: {storage_info['cost']}
                status: {storage_info['status']}
                ```"""

            self.discord.send_message(message_dc)

            """ Storage """
            try:
                self.database.store(storage_info)
            except:
                print("Failed to update")

        except ccxt.BaseError as e:
            print(f"Error creating order: {e}")

    def get_unselled_orders(self,symbol):
        cursor = self.database.collection.find({'symbol': symbol, 'sell_status': False})
        return list(cursor)

    def get_all_balance(self):
        try:
            balance_data = self.exchange.fetch_balance()

            balance = balance_data['info']['balances']
            non_zero_balance = [b for b in balance if float(b['free'])>0 or float(b['locked'])>0]
            balance_df = pd.DataFrame(non_zero_balance)
            balance_df['updateTime']=balance_data['info']['updateTime']
            balance_df['updateTime']=balance_df['updateTime'].apply(lambda x:timeTrans(int(x)))
            print(balance_df)
            balance_df.to_csv(f'balance.csv', index=False) 
            return balance_df
            
        except Exception as e:
            print(f"Error fetching balance: {e}")

    


# if __name__ == "__main__":
#     account = config(".\\config\\account.json")
#     account_info = account.load_config()
#     trade_instance = Trader(account_info)
#     #trade_instance.get_all_balance()
#     #trade_instance.get_current_positions()
#     #trade_instance.trade()
#     print(trade_instance.exchange.fetch_order("4151308",'BTC/USDT'))