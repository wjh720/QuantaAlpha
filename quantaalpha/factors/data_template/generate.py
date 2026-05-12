import qlib

from quantaalpha.utils.qlib_data import resolve_qlib_market, resolve_qlib_provider_uri, resolve_qlib_region

_provider = resolve_qlib_provider_uri()
_region = resolve_qlib_region(provider_uri=_provider)
qlib.init(provider_uri=_provider, region=_region)
from qlib.data import D

instruments = D.instruments(resolve_qlib_market(region=_region, provider_uri=_provider))
fields = ["$open", "$close", "$high", "$low", "$volume"]  # , "$amount", "$turn", "$pettm", "$pbmrq"
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2015-01-01":].sort_index()

# Calculate return
data["$return"] = data.groupby(level=0)["$close"].pct_change(fill_method=None).fillna(0)

print(data)

data.to_hdf("./daily_pv_all.h5", key="data")

fields = ["$open", "$close", "$high", "$low", "$volume"]  # , "$amount", "$turn", "$pettm", "$pbmrq"
data = (
    (
        D.features(instruments, fields, freq="day")
        .swaplevel()
        .sort_index()
    )
    .swaplevel()
    .loc[data.reset_index()["instrument"].unique()[:100]]
    .swaplevel()
    .sort_index()
)

# Calculate return
data["$return"] = data.groupby(level=0)["$close"].pct_change(fill_method=None).fillna(0)
print(data)
data.to_hdf("./daily_pv_debug.h5", key="data")
