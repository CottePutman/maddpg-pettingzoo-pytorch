import pathlib


class UnitTradeData(dict):
        def __init__(self) -> None:
            pass


class SimpleStock:
    def __init__(self, path: pathlib) -> None:
        ##TODO 通过地址加载数据
        self.unit_trade_list = None | list[UnitTradeData]
        self.open_price = None
        self.close_price = None
        
    def __len__(self) -> int:
        return len(self.unit_trade_list)
    
    def _get_open(self) -> float:
        return self.unit_trade_list[0]
    
    def _get_close(self) -> float:
         return self.unit_trade_list[1]