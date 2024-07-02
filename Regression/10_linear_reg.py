class LinearRegression:

  def __init__(learning_rate,steps,batch_size,feature_columns,
                train_x,train_y,test_x,test_y,periods_n,test_data_x,
                prediction_data,lin_reg_model,regularization_strength
                ):

    self.learning_rate = learning_rate
    self.steps = steps
    self.batch_size = batch_size
    self.feature_columns = feature_columns
    self.train_x = train_x
    self.train_y = train_y
    self.test_x = test_x
    self.test_y = test_y
    self.periods_n = periods_n
    self.test_data_x = test_data_x
    self.prediction_data = prediction_data
    self.lin_reg_model = lin_reg_model
    self.regularization_strength = regularization_strength

  def model_construct(self):
    periods_n = 10
    steps_per_period = steps / periods_n

    # Create a linear regressor object.
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                            l1_regularization_strength=regularization_strength)
    #my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )
    
    training_input_fn = lambda: my_input_fn(train_x, 
                                            train_y["target_field"], 
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(train_x, 
                                                    train_y["target_field"], 
                                                    num_epochs=1, 
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(test_x, 
                                                      test_y["target_field"], 
                                                      num_epochs=1, 
                                                      shuffle=False)

  def optimization(self):

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range (0, periods):
      # Train the model, starting from the prior state.
      linear_regressor.train(
          input_fn=training_input_fn,
          steps=steps_per_period
      )
      # Take a break and compute predictions.
      training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
      training_predictions = np.array([item['predictions'][0] for item in training_predictions])
      validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
      validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
      
      # Compute training and validation loss.
      training_root_mean_squared_error = math.sqrt(
          metrics.mean_squared_error(training_predictions, training_targets))
      validation_root_mean_squared_error = math.sqrt(
          metrics.mean_squared_error(validation_predictions, validation_targets))
      # Occasionally print the current loss.
      print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
      # Add the loss metrics from this period to our list.
      training_rmse.append(training_root_mean_squared_error)
      validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    
    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    return linear_regressor

    def predict():

    test_predictions = dnn_regressor.predict(input_fn= test_x)
    test_predictions = np.array([item['predictions'][0] for item in test_predictions])

    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_y))

    print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)


    def predict_variance(self):

      ave_pred_val = mean(validation_predictions)
      ave_pred_test = mean(test_predictions)

      variance = abs(ave_pred_val - ave_pred_test)

      return variance

    def predict_bias(self):

      ave_predictions = mean(prediction_data)
      ave_actuals = mean(test_y)

      emperical_bias = abs(ave_predictions - ave_actuals)

      return emperical_bias

    def calibration_plot_bias(self):
      # ENSURE ONLY THE POSITIVE CASES ARE ANALYZED HERE # 
      plt.figure(figsize=(10, 10))
      ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
      ax2 = plt.subplot2grid((3, 1), (2, 0))

      ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
      for clf, name in [(lin_reg_model, 'Neural Network')]:
          clf.fit(train_x, train_y)
          if hasattr(clf, "predict_proba"):
              prob_pos = clf.predict_proba(X_test)[:, 1]
          else:  # use decision function
              prob_pos = clf.decision_function(X_test)
              prob_pos = \
                  (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
          fraction_of_positives, mean_predicted_value = \
              calibration_curve(y_test, prob_pos, n_bins=10)

          ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                   label="%s" % (name, ))

          ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                   histtype="step", lw=2)

      ax1.set_ylabel("Fraction of positives")
      ax1.set_ylim([-0.05, 1.05])
      ax1.legend(loc="lower right")
      ax1.set_title('Calibration plots  (reliability curve)')

      ax2.set_xlabel("Mean predicted value")
      ax2.set_ylabel("Count")
      ax2.legend(loc="upper center", ncol=2)

      plt.tight_layout()
      plt.show()