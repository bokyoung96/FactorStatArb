from __future__ import annotations

from pre.base.align import Align
from pre.base.load import Load
from pre.base.reg import FE
from pre.factory.build import Build
from pre.factory.data import Data
import logging
import pre.price.feat_price as feat_price
import pre.fundamentals.feat_fund as feat_fund
import pre.consensus.feat_cons as feat_cons
import pre.sector.feat_sector as feat_sector

_feature_modules = (feat_price, feat_fund, feat_cons, feat_sector)

fe = list(FE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
loader = Load()
aligner = Align()
builder = Build(fe)

data = Data(loader, aligner, builder)
data.make()
