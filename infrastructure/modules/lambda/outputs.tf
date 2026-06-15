output "function_name" {
  value = aws_lambda_function.inference.function_name
}

output "function_arn" {
  value = aws_lambda_function.inference.arn
}

output "api_endpoint" {
  description = "Public HTTPS URL for the inference API"
  value       = aws_apigatewayv2_stage.default.invoke_url
}
