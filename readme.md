# Error:0

---

[팀소개](https://www.notion.so/e15309d6534d41479ffa107871a452be)

       안녕하세요?  머신러닝 모델을 구현할 때 에러를 줄이고자(error:0) 결성된 Error:0 팀입니다. 

      또한 저희조는 len(컴퓨터 공학을 전공한 사람 수)도 :0입니다.  🥸😎🤪

     다양한 전공자들이 모이다 보니, 같은 주제 해결을 위해 각자 다른 시각으로 접근할 수 있었고,

그 중 좋은 아이디어와 인사이트, 모델링 기법등을 선정해 해커톤을 진행했습니다.

 머신러닝, 딥러닝 모델이 학습을 반복할수록 Error가 줄어드는 것처럼 저희도 약 2주간의 해커톤 

기간동안 함께 학습하며 많은 시행착오 끝에 에러코드를 줄여 모델을 완성할 수 있었습니다. 

 그 결과물은 아래에 있습니다. 많은 관심 부탁드립니다. 감사합니다.

```python
print(Error:0)
[이운재, 곽민수, 김도은, 김명우, 이상재]
```

---

[주제](https://www.notion.so/bf6ee281b5d846ac9cd32f68b43fc956)

### [쏘카의 성공적인 IPO를 위한 영업이익 개선 머신러닝 모델 ]

> 소주제1 :  **쏘카존별 최적 차량 배치를 통한 이익 극대화 방안**
                 소주제2 :  **운영 종료 쏘카존 예측으로 차량 운영 효율 향상 방안**
> 

---

[개요](https://www.notion.so/00c2ab8c7f53468ba283631f749f4e9f)

### 목차

1. [데이터 수집 및 기술 이용 스택](https://www.notion.so/Error-0-1c255d1becf3468698ed989a34253a87) 
    1. 쏘카 제공 데이터
    2. 기술 이용 스택
2. [데이터 전처리](https://www.notion.so/Error-0-1c255d1becf3468698ed989a34253a87)
    1. Zone naming & indexing
    2. 시계열 변환(Timeseries)
    3. Resampling
    4. Lagging
3. [Train / Test Split](https://www.notion.so/Error-0-1c255d1becf3468698ed989a34253a87)
4. [모델링](https://www.notion.so/Error-0-1c255d1becf3468698ed989a34253a87)
    1. GradientBoostingRegressor
    2. Prophet
    3. LSTM
5. [모델 결과 분석](https://www.notion.so/Error-0-1c255d1becf3468698ed989a34253a87)
    1. 모델선정
    2. Clustering 
    3. Mapping
6. [활용방안](https://www.notion.so/Error-0-1c255d1becf3468698ed989a34253a87)
    1. 쏘카존별 최적 차량 배치
    2. 쏘카존 운영 종료 조기 예측
7. [기대효과](https://www.notion.so/Error-0-1c255d1becf3468698ed989a34253a87)
8. [회고](https://www.notion.so/Error-0-1c255d1becf3468698ed989a34253a87)

---

[내용](https://www.notion.so/0026367a0a0c4787855f465f73f553fc)

1. **데이터 수집 및 기술 이용 스택**
    1. 쏘카 제공 데이터
    저희 팀은 쏘카측에서 기본으로 제공한 2018년도 12월 31일 부터 2019년 11월 29일까지의 79만개의 데이터 셋을 최대한 이용하여 자료를 분석했습니다.
        
        (저희가 해당 raw 자료를 전처리한 파일(csv)들을 갖고 있습니다. 아래 링크 첨부하겠습니다.
        [https://drive.google.com/drive/folders/1pzGgFZt7qCsCiztEjoZvXJCJuRBntgCX?usp=sharing](https://drive.google.com/drive/folders/1pzGgFZt7qCsCiztEjoZvXJCJuRBntgCX?usp=sharing)
        
        후에 데이터 확인 원하시면 공유해드리겠습니다.)
        
        - 코드 보기
            
            ```python
            timeseries_path = os.path.join(os.getcwd(), "drive", "MyDrive",  "new_socar_data.csv")
            timeseries = pd.read_csv(timeseries_path)
            timeseries.head()
            ```
            
            ![Untitled](Error%200%206b58920a14284e338b4ccff4d551f384/Untitled.png)
            
    2. 기술 이용 스택
        
        
        | 언어 | Python |
        | --- | --- |
        | AI / ML | TensorFlow , Scikit-Learn, Prophet |
        | 기본 라이브러리 | Pandas, Numpy, matplotlib, tqdm, folium, pickle, seaborn |
        | 시각화 | Tableau |
        
2. **데이터 전처리**
    1. Zone naming & indexing
        - 서울에 있는 쏘카존
        ⇒ 전체 중 가장 존이 많이 분포되어 있는 구역 중 서울지역으로(약440여개)으로 한정했습니다.
            
            ![Untitled](Error%200%206b58920a14284e338b4ccff4d551f384/Untitled%201.png)
            
            ⇒ 또한, 이 존들 중에서 쏘카측에서 받은 데이터 상의 존들을 앞으로 저희가 다루는
            
            “쏘카존”(이하 쏘카존)이라 정의했습니다. 
            
            ⇒ 크롤링한 실제 쏘카존이 그 당시의 쏘카존의 상황과 안맞으며, 
            쏘카존이 아닌 존에서 더 많은 예약이 이루어진 것을 확인할 수 있었고, 
            크롤링한 실제 쏘카존만 추리면 머신러닝을 적용하기에 데이터가 매우 적어지기 때문입니다.
            
        - 쏘카존의 이름을 unique로 선정
        ⇒ 같은 쏘카존 주소인데 쏘카존 이름이 다른 경우가 발생하거나 같은 쏘카존 이름인데 쏘카존 주소가 다른 경우가 발생하는데 쏘카존 이름을 unique로 주었을 시의 Loss를 최소화 할 수 있기 때문입니다.
            
            <이름 value_counts() 한 값>
            
            ![Untitled](Error%200%206b58920a14284e338b4ccff4d551f384/Untitled%202.png)
            
            ![Untitled](Error%200%206b58920a14284e338b4ccff4d551f384/Untitled%203.png)
            
        
        - 코드 보기
            
            ```python
            candidates = list(timeseries["zone"].unique())
            nums = list(range(len(candidates)))
            table = {}
            for c, n in zip(candidates, nums):
              table[c] = n
            table
            timeseries.replace({"zone": table}, inplace=True)
            timeseries
            ```
            
            ![Untitled](Error%200%206b58920a14284e338b4ccff4d551f384/Untitled%204.png)
            
        
    2. 시계열 변환(Timeseries)
        - Raw 데이터로는 알 수 없는 각각 쏘카존 마다의 시간대별 수요량을 계산하기 위해 시계열 데이터로 변환했습니다.
        - 2019년 1월 1일부터 데이터가 존재하는 11월 30일까지를 1시간 단위로 쪼개 시간별 차량 운행현황을 쏘카존별로 정리했습니다. (440개존 x 8,016시간 = 352만건)
        - 차량 대여 시작과 반납 시간 정보를 이용해 대여 시간을 계산하고, 차량 ID값을 이용해 시간마다 존별로 사용중인 차량 수를 알 수 있도록 시계열 변환에 사용 했습니다.
        
         
        
        - 코드보기
            
            ```python
            last_time = pd.to_datetime("2019-01-01 00:00:00")
            hour_added = datetime.timedelta(hours = 1)
            next_time = last_time + hour_added
            month = last_time.month
            cached = seoul_data[((seoul_data["reservation_start_at"] >= last_time) & (seoul_data["reservation_start_at"] < next_time)) | ((seoul_data["reservation_return_at"] >= last_time) & (seoul_data["reservation_return_at"] < next_time)) | ((seoul_data["reservation_start_at"] < last_time) & (seoul_data["reservation_return_at"] >= next_time))]
            for index, row in tqdm(zone_timeseries.iterrows()):
              if last_time > row["time"]:
                continue
              if month != row["time"].month:
                timeseries_month_path = os.path.join(os.getcwd(), "drive", "MyDrive", f"timeseries-m.csv")
                zone_timeseries[zone_timeseries["time"].apply(lambda x: x.month) == month].to_csv(timeseries_month_path)
                print(f"month {month} is saved...")
                month = row["time"].month
              if last_time != row["time"]:
                last_time = row["time"]
                next_time = last_time + hour_added
                cached = seoul_data[((seoul_data["reservation_start_at"] >= last_time) & (seoul_data["reservation_start_at"] < next_time)) | ((seoul_data["reservation_return_at"] >= last_time) & (seoul_data["reservation_return_at"] < next_time)) | ((seoul_data["reservation_start_at"] < last_time) & (seoul_data["reservation_return_at"] >= next_time))]
              filtered = cached[cached["zone_name"] == row["zone"]]
              cached = cached[cached["zone_name"] != row["zone"]]
              zone_timeseries.loc[index, "n_drive"] = filtered.shape[0]
              zone_timeseries.loc[index, "n_drive_unique"] = len(filtered["car_id"].unique())
            
            ```
            
            ![Untitled](Error%200%206b58920a14284e338b4ccff4d551f384/Untitled%205.png)
            
        
    3. Resampling
        - 시계열 데이터로 변환된 시간마다의 차량 사용현황을 일별 & 존별 최대 예약 운행 차량 대수를 구해서 각 모델에 맞게 Resampling 구현했습니다.
        - 코드보기
            
            ```python
            ## 존별 그룹바이 한 
            timeseries["time"] = pd.to_datetime(timeseries["time"])
            ts=timeseries
            ts=ts.set_index('time')
            ts_resample= pd.DataFrame()
            ts_resample['n_drive_1Day'] = ts.groupby('zone').n_drive_unique.resample('1D').max()
            ts_resample.reset_index(inplace = True)
            ts_resample
            ```
            
            ![Untitled](Error%200%206b58920a14284e338b4ccff4d551f384/Untitled%206.png)
            
    4. Lagging
        - 시계열 데이터를 이용한 예측모델을 생성하기 위한 Lagging 작업을 진행했습니다.
        - 1일 단위로 데이터를 Resampling하고, 50~56번의 Lagging을 통해 일별 & 존별 최대 예약 운행 차량 대수 상관관계를 예측 모델에 반영하기 위해 시행했습니다.
        - 코드보기
            
            ```python
            for s in range(1,50):
              ts_resample22['shifted_{}'.format(s)] = ts_resample22.groupby('zone').n_drive_1day.shift(s+7)
            ts_resample22
            ```
            
            ![Untitled](Error%200%206b58920a14284e338b4ccff4d551f384/Untitled%207.png)
            
        
3. **Train / Test split**
    - 시계열 데이터의 범위는 2019년 1월 1일 ~ 11월 30로 전체를 9개월 : 2개월로 Split 했습니다.
    - 1월1일부터 9월 30일까지를 Train데이터로 사용해 모델에 학습을 시켰습니다.
    - 10월1일~11월 30일까지 Test 데이터로 사용해 성능 확인을 했습니다.
        - 코드보기
            
            ```python
            train = ts_resamplse22.query("time <= '2019-09-30 23:00:00'")
            test = ts_resamplse22.query("time > '2019-09-30 23:00:00'")
            x_train = np.asarray(train.drop(['n_drive_1day', 'time'],1), dtype = np.float32)
            y_train = np.asarray(train[['n_drive_1day']], dtype = np.float32)
            x_test = np.asarray(test.drop(['n_drive_1day', 'time'],1), dtype = np.float32)
            y_test = np.asarray(test[['n_drive_1day']], dtype = np.float32)
            ```
            
    
4. **모델링**
    1. GradientBoostingRegressor
        - 코드보기
            
            ```python
            # DEVIDE = zone별data행 갯수
            DEVIDE = len(ts_sample)//len(set(ts_sample["zone"].unique()))
            MSE = list()
            split_num = round(DEVIDE*0.8)
            
            # zone별로 GradientBoostingRegressor모델 생서 후 MSE 측정
            for i in range(len(ts_sample)//DEVIDE):
              d_sample = ts_sample[DEVIDE*i:DEVIDE*(i+1)]
            
              train = d_sample[:split_num]
              test = d_sample[split_num:]
              x_train = np.asarray(train.drop(['n_drive_1day','time'],1))
              y_train = np.asarray(train['n_drive_1day'])
              x_test = np.asarray(test.drop(['n_drive_1day','time'],1))
              y_test = np.asarray(test['n_drive_1day'])
            	
              reg = ensemble.GradientBoostingRegressor(n_estimators=200,max_depth=4,min_samples_leaf=1,learning_rate=0.05)
              reg.fit(x_train, y_train)
            
              mse = mean_squared_error(y_test, np.round(reg.predict(x_test)))
              MSE.append(mse)
            
              globals()['model_{}'.format(i)] = reg
            
              if i % 50 == 0:
                print("zone{} , The mean squared error (MSE) on test set: {:.4f}".format(i,mse))
            
            print()
            print("MSE average : {:.4f}".format(sum(MSE)/len(MSE)))
            
            # 각 모델을 pickle형태로 한 파일에 저장
            import pickle
            filename = "/content/drive/MyDrive/Colab Notebooks/data/socar_timeseries_models.sav"
            modlist = list()
            for i in range(len(ts_sample)//DEVIDE):
              modlist.append(globals()['model_{}'.format(i)])
            s = pickle.dump(modlist, open(filename, 'wb'))
            
            # Visualization (시각화 및 예측값 비교)
            for z in range(2):
            # z = zone number
              print(f'▼ zone{z} 실제값,예측값 비교')
              print()
              d_sample = ts_sample[DEVIDE*z:DEVIDE*(z+1)]
            
              train = d_sample[:split_num]
              test = d_sample[split_num:]
              x_train = np.asarray(train.drop(['n_drive_1day','time'],1))
              y_train = np.asarray(train['n_drive_1day'])
              x_test = np.asarray(test.drop(['n_drive_1day','time'],1))
              y_test = np.asarray(test['n_drive_1day'])
            
              plt.figure(figsize=(16,3))
              plt.grid(True)
              X = test["time"]
              Y1 = y_test
              Y2 = np.round(globals()['model_{}'.format(z)].predict(x_test))
              plt.plot(X, Y1)
              plt.plot(X, Y2,color='r') # red line = 예측값
              plt.show()
              print()
            ```
            
        
        모델링 결과 : 일자별 예측값과 실제값 비교
        
        ![1641568939975.png](Error%200%206b58920a14284e338b4ccff4d551f384/1641568939975.png)
        
    2. Prophet
        - 코드보기
            
            ```python
            period=1
            group_yhat=list()
            group_yhat_lower=[]
            group_yhat_upper=[]
            group_ds=[]
            
            y_reals = []
            y_preds = []
            zone = []
            
            index_zone=ts_resample.zone.unique()[:zone_number]
            ### 서울 전체 쏘카 존에서 시행(n=438)
            
            for i in tqdm(index_zone): 
                sample = ts_resample[ts_resample.zone ==i].copy()
            #    sample.drop('zone',axis=1,inplace=True)
            #    print(sample)
            
                if Train :
                    sample['ds'] = pd.to_datetime(sample['ds'])
                    sample.index= sample['ds']
                    sample.columns = ['ds','y']
                    
                    answer = sample.last('60D')
                    train = sample[sample['ds'] < answer['ds'].iloc[0]]
                    y_real = answer['y'].sum()
                    y_reals.append(y_real)
                else :
                    sample['ds'] = pd.to_datetime(sample['ds'])
                    sample.columns = ['zone','ds','y']
                    test= sample
                    
                m = Prophet(changepoint_prior_scale=0.8,
            
                  yearly_seasonality=False,
                  weekly_seasonality=True,
                  daily_seasonality = True,
                 # seasonality_prior_scale = 0.2,
                  holidays=holidays)
                
                m.fit(test)
                
                # 예측기간(30일로 set)
                future = m.make_future_dataframe(periods=30)    
                forecast = m.predict(future)
                new = forecast[['trend','ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period)
                new.append(new,ignore_index=True)
                group_yhat.append(new["yhat"])
                group_yhat_upper.append(new["yhat_upper"])
                group_yhat_lower.append(new["yhat_lower"])
                group_ds.append(new["ds"])
                y_pred = forecast.iloc[-30:, :].yhat.sum()
                y_preds.append(y_pred)
                zone.append(i)
            
                # 시각화부분
                fig1 = m.plot(forecast)
                plt.show()
            ```
            
        - 모델링 결과 : 예측값(파란색 실선)과 실제 Data(검은 점)
            
            ![결과.png](Error%200%206b58920a14284e338b4ccff4d551f384/%EA%B2%B0%EA%B3%BC.png)
            
        - 전체 추이 및 요일별, 시간대별 영향을 분석해 예측모델에 반영
            
            ![performance.png](Error%200%206b58920a14284e338b4ccff4d551f384/performance.png)
            
        
    3. LSTM
        - 코드보기
            
            ```python
            #일별 최대 차량 수 resampling
            timeseries_path = os.path.join(os.getcwd(), "drive", "MyDrive",  "timeseries_transformed.csv")
            timeseries = pd.read_csv(timeseries_path)
            
            timeseries["time"] = pd.to_datetime(timeseries["time"])
            ts=timeseries
            ts=ts.set_index('time')
            ts_resample= pd.DataFrame()
            ts_resample['n_drive_1Day'] = ts.groupby('zone').n_drive_unique.resample('1D').max()
            ts_resample.reset_index(inplace = True)
            ts_resample
            
            # resampling 한 데이터들을 minmaxscaler 로 정규화
            minmax_scaler = MinMaxScaler()
            sc = minmax_scaler.fit_transform(ts_resample[['n_drive_1Day']])
            
            data = pd.DataFrame(sc, columns = ['n_drive_1day'], index= ts_resample.index)
            
            data
            
            # LSTM 모델 구현
            my_LSTM_model = Sequential()
            my_LSTM_model.add(LSTM(units = 104, 
                                       return_sequences = True, 
                                       input_shape = (52,1), 
                                       activation = 'tanh'))
            my_LSTM_model.add(LSTM(units = 52, activation = 'tanh'))
            my_LSTM_model.add(Dense(units=2))
                
                # Compiling 
            my_LSTM_model.compile(optimizer = SGD(lr = 0.01, decay = 1e-7, 
                                                     momentum = 0.9, nesterov = False),
                                     loss = 'mean_squared_error')
                
                # Fitting to the training set 
            my_LSTM_model.fit(x_train, y_train, epochs = 30, batch_size = 150, verbose = 1, shuffle=False)
                
            LSTM_prediction = my_LSTM_model.predict(x_test)
            
            print(LSTM_prediction)
            
            LSTM_prediction1 = LSTM_prediction.mean(axis=1)
            LSTM_prediction1.shape
            LSTM_prediction1.reshape(26779,1)
            
            y_test2 = minmax_scaler.inverse_transform(y_test)
            LSTM_prediction2 = minmax_scaler.inverse_transform(LSTM_prediction)
            
            LSTM_prediction2
            LSTM_prediction1 = LSTM_prediction2.mean(axis=1)
            LSTM_prediction1.reshape(26779,1)
            
            mse = mean_squared_error(y_test2, LSTM_prediction1)
            mse
            
            plt.plot(y_test2, label='Origial')
            plt.plot(LSTM_prediction1, label='Prediction')
            plt.legend(loc=0)
            plt.title('Validation Results')
            plt.show()
            ```
            
        - 모델링 결과 : 일자별 예측값과 실제값 비교
            
            ![Untitled](Error%200%206b58920a14284e338b4ccff4d551f384/Untitled%208.png)
            
5. **결과 분석**
    1. 모델 선정
        - 3가지 방법의 모델링 기법을 440여개의 모든 쏘카존에 각각 Fitting해 MSE(Mean Squared Error)를 계산했습니다.
        - 그 결과, 가장 성능이 좋은 GradientBoostingRegressor 모델을 최종 선택했습니다.
        
        ※ 평균 MSE값 : **GradientBoosting : 1.08** /  **Prophet : 1.31** / LSTM : 1.38 (단위 : 대수)
        
        ![mse.png](Error%200%206b58920a14284e338b4ccff4d551f384/mse.png)
        
    2. *Clustering*
        - 선정한 모델이 예측한 쏘카존의 향후 수요를 통해 **쏘카존 별 성장 가능성 Feature를** 추출 했습니다.
        - K-Means Clustering 기법을 이용해 전체 Socar존을 4개로 구분했습니다.
        - 코드보기
            
            ![Untitled](Error%200%206b58920a14284e338b4ccff4d551f384/Untitled%209.png)
            
            ```python
            sns.lmplot(x='n_drive_avg', y='growth', data=Z, fit_reg=False,  # x-axis, y-axis, data, no line
                       scatter_kws={"s": 150}, # marker size
                       hue="cluster_id") # color
            # title
            title_font = {
                'fontsize': 16,
                'fontweight': 'bold'
            }
            plt.title('Socar Zone K-Means Clustering Result',fontdict=title_font, pad=20)
            ```
            
            ![clustering.png](Error%200%206b58920a14284e338b4ccff4d551f384/clustering.png)
            
    
    - 4개로 Clustering 된 id를 활용해 존별 특징을 분석했습니다.
        
        ※ Query, Value_counts(), groupby 등을 사용해 종합 분석했습니다.
        
        - **포인트 존**(**Point Zone,** Cluster_0)
            
            = 보유 차량의 수는 많지 않으나, 지역별로 요지에 위치하거나,
            지속적인 고정수요로 인해 꾸준한 성장을 보여주는 존 
            
        - **슈퍼 쏘카존**(**Super_Socar Zone,** Cluster_1)
            
            = 차량수요 및 운행이 많고 성장 가능성도 높아 앞으로 전망이 기대되는 존
            
        - **엔드 존**(**End Zone,** Cluster_2)
            
            = 차량수요가 적고 성장 가능성도 낮아 운영이 종료될것으로 예측되는 존
            
        - **타겟 존**(**Target Zone,** Cluster_3)
            
            = 계절 별 수요가 탄력인 곳으로 시즌별 급격한 성장 또는 감소를 보여주는 존
            
        
               ****
        
    
    c. *Mapping*
    
    - Clustering된 쏘카 존들을 지도상에 표시해 지리적 특징을 파악했습니다.
    - 코드보기
        
        ```python
        map_osm = folium.Map(location=[37.5, 127], zoom_start=12)
        
        for index, row in z_name_total.iterrows():
            location = (row['zone_lat'], row['zone_lng'])
         
            if row['cluster_id']==3:
              folium.Marker(location, popup=row['zone'],icon = folium.Icon(color='purple', icon='circle')).add_to(map_osm)
            if row['cluster_id']==2:
              folium.Marker(location, popup=row['zone'],icon = folium.Icon(color='gray', icon='circle')).add_to(map_osm)
            if row['cluster_id']==1:
              folium.Marker(location, popup=row['zone'],icon = folium.Icon(color='orange', icon='star')).add_to(map_osm)
            if row['cluster_id']==0:
              folium.Marker(location, popup=row['zone']).add_to(map_osm)
        
        map_osm
        ```
        
    
    ![mapping.png](Error%200%206b58920a14284e338b4ccff4d551f384/mapping.png)
    
6. **활용방안**
    1. **쏘카존별 최적 차량 배치**
        
        ※ Target Zone은 계절별 수요가 탄력적이므로 수요가 없는 시즌에는 Super_Socar Zone 으로
            차량을 이동하고, 수요가 높은 시기는 Target존에 차량을 집중하는 방법이 효과적입니다. 
        
        - 지역의 Target 또는 Super_Socar Zone 과 다른 쏘카존 사이의 상관관계(Correlation)를 계산해 인근 지역에서 수요가 반대로 움직이는 존들을 Grouping 했습니다.
        
          ▶ 수요가 반대로 움직이는 존 : 특정 존의 차량 수요가 늘어남에 따라 수요가 떨어지는 존
        
        ![Untitled](Error%200%206b58920a14284e338b4ccff4d551f384/Untitled%2010.png)
        
        - Grouping한 쏘카존들의 유휴차량 Data를 Timeline으로 분석해 특정 시점의 존당 유휴차량과 부족차량을 판단했습니다.
        - 유휴차량이 많은 존에서 차량이 부족한 존으로의 차량 재배치를 실시할 수 있습니다.
        - 따라서 이런 regression 기법을 통해 슈퍼 쏘카존과 타겟존 사이의 시간대별 비운행 차량을 예측해 비운행 차량을 줄여 줄 수 있는 모델을 만들어 보았습니다
        
        ※ **AJ파크 논현점, e편한세상 4단지 AJ파크, 풍산빌딩** 존의 유휴차량 수 타임라인 분석
        
        ![3d_corr.png](Error%200%206b58920a14284e338b4ccff4d551f384/3d_corr.png)
        
        ⇒ 5월초의 경우 **e편한세상 4단지 AJ파크, 풍산빌딩**존은 차량이 부족했으나, 인근 **AJ파크 논현점의 경우 차량을 충분히 보유하고 있었습니다.** 
        
        ⇒ 모델이 예측한 결과 12월의 경우 **e편한세상 4단지 AJ파크**의 차량을 **풍산빌딩** 또는 **AJ파크로 재배치 할 필요가 있다**고 보입니다. 
        
    2. **쏘카존 운영 종료 시점 예측**
        - 차량수가 많지 않은 Point Zone과 End Zone을 대상으로 Classification 모델을 생성해 쏘카존의 운행종료 시점을 **약 95% 정확도**로 조기 예측했습니다.
            
            ```python
            from sklearn.ensemble import GradientBoostingClassifier
            
            gb1 = GradientBoostingClassifier(random_state=0, max_depth=2) # by deafult 3
            gb1.fit(x_train, y_train)
            
            print("Accuracy on training set: {:.3f}".format(gb1.score(x_train, y_train)))
            print("Accuracy on test set: {:.3f}".format(gb1.score(x_test, y_test)))
            ```
            
            ![Untitled](Error%200%206b58920a14284e338b4ccff4d551f384/Untitled%2011.png)
            
        - 존별 데이터를 Timeline으로 분석해 수요가 줄어들것으로 예상되는 시점과 그 트렌드 분석을 통한 존별 운행 종료 시점을 조기 판단할 수 있었습니다.
        - 기존 운행종료로 판단된 존들과 비교했을때 약 **2달정도 빠른 예측**을 보였습니다.
        - 시계열과 기존 Raw데이터를 연계해 Classification 결과를 시각적으로 표시했습니다.
            
            ※ 운행종료 예측모델 시각화 (존별 차량 운행 현황을 월별로 분석, 운행일수 표시) 
            
            ![closed.png](Error%200%206b58920a14284e338b4ccff4d551f384/closed.png)
            
        
7. **기대효과** 
    
    위 활용방안을 통해 약 **52.1억원의 재무 개선 효과**가 있을 것으로 예상됩니다. 
    
    이는 2019년 회계 기준 영업손실액인 716억 대비 약  **7.28%가 개선**된 수치입니다. 
    
    
    1) 쏘카존별 최적 차량 배치를 통한 수익 극대화로 **31.9억원 영업이익 증가**
    
      ※ 산출근거 : 존별 유휴차량 0 인 시간 평균/년 x 시간당 차량 대여가격 x 전국 쏘카 존수
    
    > 99.64 x 8,000원 x 4000 = 3,188,480,000원
    
    
    
    2) 차량 운영 효율 향상으로 차량 고정비 및 감가상각 등 **4.2억원 비용절감**
    
      ※ 산출근거 : 효율향상 기간 x 예상 운영종료 존 수 x (감가상각비+고정비)
    
    > 2Month x 200(전국기준) x (0.2억 x 1/36+0.005억) = 4.2억원
    
    
    
    3) 운영종료 존 조기 결정을 통한 차량 수익향상 으로 **영업이익 16억원 증가**
    
      ※ 산출근거 : 효율향상 기간 x 차량 운행시간 증가분/월 x 차량 대여가격 x 종료 존 차량수
    
      ※ 산출근거 : 효율향상 기간 x 예상 운영종료 존 수 x (감가상각비+고정비)
    
    > 2Month x 50H/월 x 8,000원/H x 2,000대 (전국기준) = 16억원
    
    
8. **회고**
    
    쏘카는 매년 성장중이며, 2021년 3분기 흑자 전환했습니다. 저희가 개발한 머신러닝 모델을 통해
    
    쏘카의  재무상황이 더욱 개선되어 IPO에서 충분한 밸류에이션을 인정받기를 바랍니다. 
    
    그 순간을 외부가 아닌 내부에서 함께 할 수 있으면 좋겠습니다. 
    
    이번 해커톤을 진행하면서 데이터와 시간의 부족함을 느껴 저희 팀명처럼 에러를 더 줄일 수
    
    있는 머신러닝 모델을 구현하고자 하는 욕구가 생겼습니다. 기회가 된다면 아래의 정보 등도
    
    반영해 모델의 성능을 더 향상시키고 싶습니다.
    
    [Time Series From Scratch - Exponentially Weighted Moving Averages (EWMA) Theory and Implementation](https://towardsdatascience.com/time-series-from-scratch-exponentially-weighted-moving-averages-ewma-theory-and-implementation-607661d574fe)
    
    P.S. 귀중한 데이터를 제공해준 쏘카 데이터 그룹장님과 직원분들, 훌륭한 강의와 커리큘럼을
           통해 AI엔지니어의 길을 열어주신 강사님과 매니저님께도 감사드립니다.
    

[참고자료](https://www.notion.so/e3d2d7df847e4ef2bf6fe6a4a0fb88c4)

[Prophet](https://facebook.github.io/prophet/)

[쏘카 실전 데이터로 배우는 AI 엔지니어 육성 부트캠프](https://classlion.net/class/detail/43)

[](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/)

[Basic Feature Engineering With Time Series Data in Python - Machine Learning Mastery](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/)

[㈜쏘카 기업정보 - 연봉 4,327만원 | 잡코리아](https://www.jobkorea.co.kr/company/16152121)

[쏘카/연결감사보고서/2021.04.02](https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20210402000328)
