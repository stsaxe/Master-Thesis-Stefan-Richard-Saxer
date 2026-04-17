import unittest

from evaluation_framework import StaticMetricCollection, DynamicMetricCollection
from evaluation_framework.metrics.closed_set.accuracy import Accuracy
from evaluation_framework.metrics.closed_set.accuracy_topk import TopKAccuracy
from evaluation_framework.metrics.container.metric_container import MetricContainer


class TestMetricCollection(unittest.TestCase):
    def test_static_collection(self):
        acc = Accuracy()

        config = StaticMetricCollection({'Acc': acc,
                                         'Top-2': TopKAccuracy(topK=2)})

        metrics = config.get_metrics()
        self.assertIsInstance(metrics[0], MetricContainer)
        self.assertEqual(metrics[1].get_name(), 'Top-2')
        self.assertNotEqual(id(acc), id(metrics[0]))
        self.assertEqual(len(metrics), 2)

    def test_static_collection_list(self):
        acc = Accuracy()

        config = StaticMetricCollection([Accuracy(average='macro'),
                                          TopKAccuracy(topK=2)])

        metrics = config.get_metrics()
        self.assertIsInstance(metrics[0], MetricContainer)
        self.assertEqual(metrics[1].get_name(), 'Top-2 Accuracy')
        self.assertEqual(metrics[0].get_name(), 'Accuracy (macro)')
        self.assertNotEqual(id(acc), id(metrics[0]))
        self.assertEqual(len(metrics), 2)

    def test_dynamic_collection(self):
        # test adding to collection
        acc = Accuracy()

        config = DynamicMetricCollection({'Acc': acc,
                                          'Top-2': TopKAccuracy(topK=2)})

        metrics = config.get_metrics()
        self.assertIsInstance(metrics[0], MetricContainer)
        self.assertEqual(metrics[1].get_name(), 'Top-2')
        self.assertNotEqual(id(acc), id(metrics[0]))

        self.assertEqual(len(metrics), 2)

        metrics = {
            'Top-3': TopKAccuracy(topK=3),
            'Top-3-V2': TopKAccuracy(topK=3)
        }

        config.add_metrics(metrics)
        metrics = config.get_metrics()

        self.assertIsInstance(metrics[3], MetricContainer)
        self.assertEqual(len(metrics), 4)

        more_metrics = [TopKAccuracy(topK=10)]
        config.add_metrics(more_metrics)

        config.add_metric(TopKAccuracy(5), 'Top 5 Metric')
        config.add_metric(TopKAccuracy(6))

        metrics = config.get_metrics()
        self.assertEqual(len(metrics), 7)

        for metric in metrics:
            self.assertIsInstance(metric, MetricContainer)

        self.assertEqual('Top-10 Accuracy', metrics[-3].get_name())
        self.assertEqual('Top 5 Metric', metrics[-2].get_name())
        self.assertEqual('Top-6 Accuracy', metrics[-1].get_name())

        # test removeing from collection
        config.remove_metric('Top 5 Metric')
        metrics = config.get_metrics()
        self.assertEqual('Top-10 Accuracy', metrics[-2].get_name())

        config.remove_metrics(['Top-3', 'Top-3-V2'])
        metrics = config.get_metrics()
        self.assertEqual(len(metrics), 4)
        self.assertEqual('Top-10 Accuracy', metrics[2].get_name())

        # test assertions
        with self.assertRaises(AssertionError) as e:
            config.add_metric(Accuracy(), 'Acc')

        with self.assertRaises(AssertionError) as e:
            config.add_metrics([123])

        with self.assertRaises(AssertionError) as e:
            config.add_metrics([123])

        with self.assertRaises(AssertionError) as e:
            config.add_metrics({'Metric': 123})

        with self.assertRaises(AssertionError) as e:
            config.remove_metric('Metric')

        with self.assertRaises(AssertionError) as e:
            config.remove_metrics(['Top-10 Accuracy', 'Test'])

        with self.assertRaises(AssertionError) as e:
            config.remove_metrics({'Top-10 Accuracy'})

        with self.assertRaises(AssertionError) as e:
            config.remove_metrics(['Top-10 Accuracy', 123])






