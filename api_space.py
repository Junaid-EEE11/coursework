url="https://api.spacexdata.com/v4/launches/past"
response=request.get(url)

data=pd.normalize(response.json())
getBoosterVersion(data)
